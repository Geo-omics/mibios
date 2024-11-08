from collections import Counter
from logging import getLogger

from django.conf import settings
from django.contrib.contenttypes.models import ContentType
from django.contrib.postgres.search import \
    SearchQuery, SearchRank, TrigramDistance
from django.db import connections
from django.db.models import Count, F, Func, Q, TextField, Value, Window
from django.db.models.functions import FirstValue, Length, RowNumber

import pandas

from mibios.omics.queryset import SampleQuerySet as OmicsSampleQuerySet
from mibios.umrad.manager import QuerySet
from . import GREAT_LAKES
from .search_utils import SearchResult
from .utils import split_query


log = getLogger(__name__)


class DatasetQuerySet(QuerySet):

    _allowed_pks = {}
    """ class-level cache, PKs that a user is allowed """

    @classmethod
    def _get_allowed_pks(cls, model, user):
        if user not in cls._allowed_pks:
            q = Q(restricted_to=None)  # no restrictions
            if user.is_authenticated:
                # or restriction to a user's group
                q = q | Q(restricted_to__user=user)

            qs = model.objects.filter(q).values_list('pk', flat=True)
            cls._allowed_pks[user] = tuple(qs)
        return cls._allowed_pks[user]

    @classmethod
    def clear_allowed_pks(cls, **kwargs):
        """
        Clear the allowed PKs cache

        Should be called whenever the Dataset.restricted_to relation changes
        """
        cls._allowed_pks = {}

    def get_allowed_pks(self, user):
        # this is a wrapper just to inject the model into the class method
        return self._get_allowed_pks(self.model, user)

    def exclude_private(self, user):
        """
        Exclude private datasets for which user is not member of allowed group.
        """
        return self.filter(pk__in=self.get_allowed_pks(user))

    def summary(
            self,
            column_field='sample_type',
            row_field='geo_loc_name',
            as_dataframe=True,
            otherize=True,
    ):
        """
        Compute basic count statistics for front-page

        Parameters:  column_field and row_fields must be char field names from
        the Sample model.

        Datasets are counted multiples times if they contain samples from
        several lakes or of more than one type.
        """
        if otherize and not as_dataframe:
            raise RuntimeError('otherize implies as_dataframe')

        col_field = 'sample__' + column_field
        row_field = 'sample__' + row_field
        qs = (self
              .values_list('pk', row_field, col_field)
              .order_by(row_field, col_field)
              .distinct()
              )

        # When counting below, do not account for missing type (b), hence we
        # test for b, missing location (a) is OK as those normally go into
        # 'other'
        counts = Counter(((a, b) for _, a, b in qs if b))

        if not as_dataframe:
            return counts

        counts = pandas.Series(
            counts.values(),
            index=pandas.MultiIndex.from_tuples(
                counts.keys(),
                names=(row_field, column_field),
            ),
            dtype=int,
        )
        df = counts.unstack(fill_value=0)

        if otherize:
            greats = set(GREAT_LAKES)
            others = df.loc[[i for i in df.index if i not in greats]].sum()
            df = df.loc[df.index.intersection(greats)]
            if df.columns.empty:
                # no datasets in DB, assigning others would fail with
                # "ValueError: cannot set a frame with no defined columns"
                pass
            else:
                df.loc['other'] = others

        return df


class SampleQuerySet(OmicsSampleQuerySet):

    _allowed_pks = {}

    @classmethod
    def _get_allowed_pks(cls, model, user):
        if user not in cls._allowed_pks:
            Dataset = model._meta.get_field('dataset').related_model
            qs = model.objects.filter(
                dataset__in=Dataset.objects.exclude_private(user)
            )
            qs = qs.values_list('pk', flat=True)
            cls._allowed_pks[user] = tuple(qs)
        return cls._allowed_pks[user]

    def get_allowed_pks(self, user):
        # this is a wrapper just to inject the model into the class method
        return self._get_allowed_pks(self.model, user)

    @classmethod
    def clear_allowed_pks(cls, **kwargs):
        """
        Clear the allowed PKs cache

        Should be called whenever the Dataset.restricted_to relation changes
        """
        cls._allowed_pks = {}

    def exclude_private(self, user):
        """
        Exclude samples of private datasets unless user is member of allowed
        group.
        """
        return self.filter(pk__in=self.get_allowed_pks(user))

    def basic_counts(self, *fields, exclude_blank_fields=[]):
        """
        Count samples by given combination of categories
        """
        if not fields:
            raise ValueError('fields parameter missing')
        qs = self
        for i in exclude_blank_fields:
            qs = qs.exclude(**{i: None})
        qs = (qs
              .values_list(*fields)
              .order_by(*fields)
              .annotate(count=Count('*'))
              )
        return qs

    def summary(
        self,
        column_field='sample_type',
        row_field='geo_loc_name',
        blank_sample_type=False,
        otherize=True,
    ):
        """ Get count summary (for frontpage view) """
        if not blank_sample_type:
            if 'sample_type' in [column_field, row_field]:
                # with default paramters: don't display stats for samples with
                # missing sample type
                exclude_blank_fields = ['sample_type']
        else:
            exclude_blank_fields = []

        qs = self.basic_counts(row_field, column_field,
                               exclude_blank_fields=exclude_blank_fields)

        # It's not totally trivial to turn the QuerySet, a sequence of tuples
        # (row_name, column_name, count) into a DataFrame.  So we put the
        # counts into a Series and make a multi-index from the
        # row/column-names.  Then we need to pivot this, but keep it integer
        # and fill missing values with zero (which unstack() can do with a
        # multi index)
        counts = pandas.Series(
            (count for _, _, count in qs),
            index=pandas.MultiIndex.from_tuples(
                ((a, b) for a, b, _ in qs),
                names=(row_field, column_field),
            ),
        )
        df = counts.unstack(fill_value=0)

        if otherize and row_field == 'geo_loc_name':
            greats = set(GREAT_LAKES)
            others = df.loc[[i for i in df.index if i not in greats]].sum()
            df = df.loc[df.index.intersection(greats)]
            if df.columns.empty:
                # no datasets in DB, assigning others would fail with
                # "ValueError: cannot set a frame with no defined columns"
                pass
            else:
                df.loc['other'] = others

        return df


class ts_headline(Func):
    """
    Postgresql's ts_headline function for search result highlighting

    This functions receives three arguments:
        * the document/text which was hit by the query
        * the tsquery which we can provide with a SearchQuery(query,
          search_type)
        * options string, a str with a ,-separated list of key=val options
    """
    function = 'ts_headline'
    arity = 3
    output_field = TextField()


class SearchableQuerySet(QuerySet):

    _model_cache = None
    DEFAULT_HARD_LIMIT = 1000

    @classmethod
    def get_model_cache(cls):
        if cls._model_cache is None:
            cls._model_cache = {
                i.pk: i.model_class()
                for i in ContentType.objects.all()
                if i.model_class() is not None  # stale contenttype
            }
        return cls._model_cache

    @classmethod
    def get_content_type_ids(cls, *model_names):
        """
        Helper to get content type PKs for given models
        """
        pks = []
        for pk, model in cls.get_model_cache().items():
            if model._meta.model_name in model_names:
                pks.append(pk)
        if len(pks) == len(model_names):
            return pks
        else:
            raise ValueError('some given model names do not correspond to a '
                             'content type')

    def search_qs(
        self,
        query,
        abundance=False,
        models=[],
        fields=[],
        lookup=None,
        search_type='websearch',
        highlight=None,
        user=None,
    ):
        """
        Helper for search().  Returns the queryset.

        :param bool abundance:
            If True, then results are limited to those with abundance / related
            genes in dataset etc.
        :param list models:
            list of str of model names, limit search to given models
        :param list fields:
            list of field names, limits results to these fields
        :param str lookup:
            Use this lookup to query the text field.  If this is None, then
            default to 'icontains' on sqlite.  On postgres the default is to do
            full text search.
        :param str search_type:
            The type of search for postresql's full text search.
        """
        # use postgres full-text search if possible
        pg_textsearch = (
            connections[self.db].vendor == 'postgresql'
            and lookup is None
        )

        if highlight is None:
            # by default, do highlighting with pg full text search
            highlight = pg_textsearch

        f = {}
        if abundance:
            f['has_hit'] = True
        if models:
            f['content_type_id__in'] = self.get_content_type_ids(*models)
        if fields:
            f['field__in'] = fields
            qs = self

        if pg_textsearch:
            tsquery = SearchQuery(query, search_type=search_type)
            f['searchvector'] = tsquery
        else:
            # sqlite etc. or specific lookup requested
            if lookup is None:
                # set default lookup for sqlite use
                lookup = 'icontains'
            f[f'text__{lookup}'] = query

        qs = self.filter(**f)

        if user:
            if not models or 'dataset' in models:
                is_dataset = Q(content_type_id=self.get_content_type_ids('dataset')[0])  # noqa:E501
                dataset_allowed = Q(dataset__restricted_to=None)
                if user.is_authenticated:
                    dataset_allowed |= Q(dataset__restricted_to__user=user)
                qs = qs.filter(~is_dataset | dataset_allowed)

            if not models or 'sample' in models:
                is_sample = Q(content_type_id=self.get_content_type_ids('sample')[0])  # noqa:E501
                sample_allowed = Q(sample__dataset__restricted_to=None)
                if user.is_authenticated:
                    sample_allowed |= Q(sample__dataset__restricted_to__user=user)  # noqa:E501
                qs = qs.filter(~is_sample | sample_allowed)

        if pg_textsearch and highlight:
            qs = qs.annotate(snippet=ts_headline(
                F('text'),
                tsquery,
                Value('StartSel=<mark>, StopSel=</mark>'),
            ))

        qs = qs.order_by('content_type_id')
        return qs

    def rank(self):
        """
        rank the full-text search results

        This will only do anything interesting for postgresql queries filtered
        by a SearchQuery.  Otherwise the same arbitrary default rank is
        assigned to every hit.

        Results get ordered (secondarily) by rank s.t. existing order is
        preserved.
        """
        # Extract the SearchQuery from the search_qs() call or equivalent
        # filter().
        tsqueries = [
            i.rhs for i in self.query.where.children
            if isinstance(getattr(i, 'rhs', None), SearchQuery)
        ]
        if len(tsqueries) == 1:
            rank = SearchRank(F('searchvector'), tsqueries[0])
            qs = self.annotate(rank=rank)
            return qs.order_by(*qs.query.order_by, '-rank')
        else:
            return self.annotate(rank=Value(0))

    def limit(self, start=1, limit=None):
        """
        Limit result set per model

        start: The first row to return, counting from 1.
        limit: how many rows to return, if None then return all rows.

        This depends on an order by content_type.

        Return type is a RawQuerySet so calling other queryset methods on this
        may be tricky.
        """
        if start <= 1 and limit is None:
            return self

        if not isinstance(limit, int):
            raise ValueError('If not None, then limit must be an integer')

        if self.query.order_by and self.query.order_by[0] == 'content_type_id':
            pass
        else:
            raise RuntimeError('do not call limit() unless you called '
                               'order_by("content_type_id", ...) beforehand')

        # For Django <= 4.2? (where they added filtering on window function
        # annotations), we do this subquery/raw dance.  No idea how to do this
        # without raw()
        qs = self.annotate(
            rownum=Window(RowNumber(), partition_by=F('content_type_id'))
        )
        sub_sql, params = qs.query.sql_with_params()
        if 1 < start and limit is None:
            cond = '%s <= rownum'
            params += (start,)
        elif 1 < start:
            cond = '%s <= rownum and rownum < %s'
            params += (start, start + limit)
        else:
            # start == 1 with limit
            cond = 'rownum <= %s'
            params += (limit,)
        sql = f'SELECT * FROM ({sub_sql}) as foo where {cond}'
        return self.model.objects.raw(sql, params)

    def search(self, query, highlight=None, soft_limit=None,
               hard_limit=DEFAULT_HARD_LIMIT,
               **kwargs):
        """
        Full-text search

        This, together with fallback_search() is the main entry point to FTS.

        :param str query: The search string.
        :param bool highlight:
            Set to True to HTML-highlight the query term in hits.  If this is
            None / not set, then this defaults to whether highlighting is
            implemented, currently only with postgresql full text search.
        :param int soft_limit: how many hits (per model) to return (to user)
        :param int hard_limit:
            the (per-model) limit of rows to ask the DB for, we use this to
            gauge the total number of hits.
        :param dict kwargs: Parameters passed on to search_qs().

        Returns a SearchResult object.
        """

        qs = self.search_qs(query, highlight=highlight, **kwargs)
        qs = qs.rank()
        qs = qs.limit(limit=hard_limit)

        return SearchResult.from_searchables(
            qs,
            self.get_model_cache(),
            soft_limit=soft_limit,
            hard_limit=hard_limit,
        )

    def fallback_search(self, query, force=False, search_type='websearch',
                        **kwargs):
        """
        Run a less restrictive search

        Assume you ran search() but got no results.  This method with the same
        signature as search() plus the force kwarg will try to run a less
        restrictive search by converting all &s to |s in the tsquery (all
        conjunctions to disjunctions).  For some simple queries, e.g. a single
        word this won't make any difference and an empty result is returned
        without querying the database.  For the phrase search type the database
        is never queried because its tsquery string has never any &s.  This
        shortcutting behavior can be overridden by force=True which always
        triggers a database query.  The raw search type is not supported as its
        tsquery can have arbitrary complex structure and the conversion can not
        be done with a simple str.replace().

        Also this will only work in postgresql.
        """
        try:
            tsquery = self.model.objects.tsquery_from_str(query, search_type)
        except Exception as e:
            if settings.DEBUG:
                raise
            else:
                log.error(f'error getting tsquery for fallback search: '
                          f'{e.__class__.__name__}: {e}')
                return SearchResult.empty()

        if tsquery:
            try:
                new_tsquery = self.model.objects.get_fallback_tsquery(tsquery)
            except Exception as e:
                if settings.DEBUG:
                    raise
                else:
                    log.error(f'error transforming to fallback tsquery'
                              f'{e.__class__.__name__}: {e}')
                    return SearchResult.empty()
        else:
            # query was some nonsense, e.g. a single letter and postgres gave
            # us an empty tsquery.
            new_tsquery = None

        log.debug(f'FALLBACK SEARCH {query=} {tsquery=} {new_tsquery=}')
        if not tsquery or (new_tsquery == tsquery and not force):
            # shortcut, assumes original query failed, return empty result
            return SearchResult.empty()
        else:
            return self.search(new_tsquery, search_type='raw', **kwargs)


class UniqueWordQuerySet(QuerySet):
    DISTANCE_CUTOFF = 0.8
    """ don't suggest spellings further away than this """

    def suggest_word(self, word, always=False, check_length=1, limit=20):
        """
        Suggest spelling for a single word

        Returns a QuerySet.

        If the given word is correctly spelled it is returned alone, without
        further suggestions, unless the always=True option is used.
        """
        qs = self.suggest_word_always(
            word,
            check_length=check_length,
            limit=limit,
        )
        if always:
            return qs

        # If we have an exact match, then return only that, not anything else.
        # This is almost super clever, it needs raw('...WHERE...') because (I
        # think) the window expression annotation may not appear in a
        # subsequent filter.  Otherwise we would just add
        # filter(least_dist__gt=0.0) (some Django limitation.)
        qs = qs.annotate(least_dist=Window(expression=FirstValue('dist')))
        qs_sql, params = qs.query.sql_with_params()
        qs = self.model.objects.raw(
            f'SELECT * FROM ({qs_sql}) _ WHERE dist = 0.0 OR least_dist > 0.0',
            params,
        )
        return qs

    CHECK_LENGTHS = {
        5: 1,
        10: 4,
    }

    def suggest_word_always(self, word, check_length=1, limit=20):
        """
        Suggest spelling for a single word

        This method always returns variations, even if word is a perfect match.
        """
        qs = self.annotate(dist=TrigramDistance('word', word))
        qs = qs.filter(dist__lte=self.DISTANCE_CUTOFF)

        if check_length is None:
            # Allow less length variation for short words (about +1/-1 seems
            # right.  For very long, unknown words there are often good matches
            # among the shorter vocabulary,
            for cutoff, check_length in self.CHECK_LENGTHS.items():
                if len(word) >= cutoff:
                    break
            else:
                check_length = int(len(word) / 1.5)

        if isinstance(check_length, int):
            if check_length >= 0:
                qs = qs.annotate(length=Length('word')) \
                    .filter(
                        length__lte=len(word) + check_length,
                        length__gte=len(word) - check_length,
                )
        elif check_length:
            raise ValueError('expect an int or evaluate to False')

        qs = qs.order_by('dist', 'word')

        if limit:
            return qs[:limit]
        else:
            return qs

    def suggest_phrase(self, txt, check_length=1, limit=20):
        """
        Suggest spelling for a phrase

        Returns a dict mapping words (in original order) to list of closest
        matches.  Matches are tuples (distance, word) and are sorted from small
        to large distance.  An empty list indicates that the word wasn't found
        and no spelling is suggested, possibly because of distance cutoff.  A
        None value means correct spelling.
        """
        if isinstance(txt, str):
            auto_mode = True
            txt = split_query(txt, keep_quotes=True)
        else:
            auto_mode = False

        suggestions = {}
        for word in txt:
            if auto_mode and word.startswith(('-', "'", '"')):
                # auto mode: don't check quoted text or negated words
                # Same as if spelled correctly.
                suggestions[word] = None
                continue
            matches = list(self.suggest_word(
                word,
                always=False,
                check_length=check_length,
                limit=limit,
            ))
            if not matches:
                suggestions[word] = []
            elif matches[0].dist == 0.0:
                # spelled correctly
                suggestions[word] = None
            else:
                suggestions[word] = [(i.dist, i.word) for i in matches]

        return suggestions

    def suggest(self, txt, check_length=None, limit=20):
        """
        Get spelling suggestions

        :params int limit: Limit to thois many suggestions (per word!)

        This is the main entry-point for the whole spelling suggestion feature,
        usually called on the unfiltered table, as in
        UniqueWord.objects.all().suggest(...).
        """
        suggestions = self.suggest_phrase(
            txt,
            check_length=check_length,
            limit=limit,
        )

        return {
            # suggestions w/o distance
            word: None if lst is None else [i for _, i in lst]
            for word, lst in suggestions.items()
        }
