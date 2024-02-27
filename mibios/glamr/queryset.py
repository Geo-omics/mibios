from collections import Counter
from itertools import groupby
from operator import attrgetter

from django.contrib.postgres.search import SearchQuery, TrigramDistance
from django.db import connections
from django.db.models import Count, F, Func, TextField, Value, Window
from django.db.models.functions import FirstValue, Length
from django.utils.safestring import mark_safe

import pandas

from mibios.omics.queryset import SampleQuerySet as OmicsSampleQuerySet
from mibios.umrad.manager import QuerySet
from . import GREAT_LAKES, HORIZONTAL_ELLIPSIS
from .utils import split_query


class DatasetQuerySet(QuerySet):
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

    This functions recevies three arguments:
        * the document/text which was hit by the query
        * the tsquery which we can provide with a SearchQuery(query,
          search_type)
        * options string, a str with a ,-separated list of key=val options
    """
    function = 'ts_headline'
    arity = 3
    output_field = TextField()


class SearchableQuerySet(QuerySet):
    def search_qs(
        self,
        query,
        abundance=False,
        models=[],
        fields=[],
        lookup=None,
        search_type='websearch',
        highlight=None,
    ):
        """
        Helper for search().  Returns the qeryset.

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
            f['content_type__model__in'] = models
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

        qs = self.filter(**f) \
            .select_related('content_type') \
            .order_by('content_type', 'field')

        if pg_textsearch and highlight:
            qs = qs.annotate(snippet=ts_headline(
                F('text'),
                tsquery,
                Value('StartSel=<mark>, StopSel=</mark>'),
            ))
        return qs

    def search(self, query, highlight=None, **kwargs):
        """
        Full-text search

        :param str query: The search string.
        :param bool highlight:
            Set to True to HTML-highlight the query term in hits.  If this is
            None / not set, then this defaults to whether highlighting is
            implemented, currently only with postgresql full text search.
        :param dict kwargs: Parameters passed on to search_qs().

        Returns a dict mapping models to dict mapping PKs to "snippets."
        Snippets are lists of (field, text) tuples.
        """
        qs = self.search_qs(query, highlight=highlight, **kwargs)

        result = {}
        for ctype, model_hits in groupby(qs, key=attrgetter('content_type')):
            model = ctype.model_class()
            result[model] = {}
            key = attrgetter('object_id')
            for object_id, obj_hits in groupby(model_hits, key=key):
                result[model][object_id] = []
                for i in sorted(obj_hits, key=attrgetter('field')):
                    if snippet := getattr(i, 'snippet', None):
                        # ts_highlight() individually marks consequtive words
                        # of a matching phrase, let's highlight the whole
                        # phrase
                        snippet = snippet.replace('</mark> <mark>', ' ')
                        # add ... if snippet is inside the text
                        plain = snippet.replace('<mark>', '').replace('</mark>', '')  # noqa:E501
                        if not i.text.startswith(plain):
                            snippet = f'{HORIZONTAL_ELLIPSIS} {snippet}'
                        if not i.text.endswith(plain):
                            snippet = f'{snippet} {HORIZONTAL_ELLIPSIS}'
                    else:
                        # highlighting is OFF
                        snippet = i.text

                    snippet = mark_safe(snippet)
                    result[model][object_id].append((i.field, snippet))

        return result


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
