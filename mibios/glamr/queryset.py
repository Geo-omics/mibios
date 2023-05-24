from collections import Counter
from itertools import groupby
from operator import attrgetter

from django.contrib.postgres.search import SearchQuery, TrigramDistance
from django.db import connections
from django.db.models import Count, Window
from django.db.models.functions import FirstValue, Length

import pandas

from . import GREAT_LAKES
from mibios.umrad.manager import QuerySet


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
        )
        df = counts.unstack(fill_value=0)

        if otherize:
            greats = set(GREAT_LAKES)
            others = df.loc[[i for i in df.index if i not in greats]].sum()
            df = df.loc[GREAT_LAKES]
            df.loc['other'] = others

        return df


class SampleQuerySet(QuerySet):
    def basic_counts(self, *fields, exclude_blanks=True):
        """
        Count samples by given combination of categories
        """
        if not fields:
            raise ValueError('fields parameter missing')
        qs = self
        if exclude_blanks:
            for i in fields:
                qs = qs.exclude(**{i: ''})
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
        exclude_blanks=True,
        otherize=True,
    ):
        """ Get count summary (for frontpage view) """
        qs = self.basic_counts(row_field, column_field,
                               exclude_blanks=exclude_blanks)

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

        if otherize:
            greats = set(GREAT_LAKES)
            others = df.loc[[i for i in df.index if i not in greats]].sum()
            df = df.loc[GREAT_LAKES]
            df.loc['other'] = others

        return df


class SearchableQuerySet(QuerySet):
    def search(
        self,
        query,
        abundance=False,
        models=[],
        fields=[],
        lookup=None,
    ):
        """
        Full-text search

        :param bool abundance:
            If True, then results are limited to those with abundance / related
            genes in dataset etc.
        :param list models:
            list of str of model names, limit search to given models
        :param list fields:
            list of field names, limits results to these fields
        :param str lookup: Use this lookup to query the text field
        """
        f = {}
        if abundance:
            f['has_hit'] = True
        if models:
            f['content_type__model__in'] = models
        if fields:
            f['field__in'] = fields
            qs = self

        # use postgres full-text search if possible
        if connections[self.db].vendor == 'postgresql' and lookup is None:
            f['searchvector'] = SearchQuery(query, search_type='websearch')
        else:
            # sqlite etc. or specific lookup requested
            if lookup is None:
                # set default lookup for sqlite use
                lookup = 'icontains'
            f[f'text__{lookup}'] = query

        qs = self.filter(**f) \
            .select_related('content_type') \
            .order_by('content_type', 'field')

        result = {}
        for ctype, out_grp in groupby(qs, key=attrgetter('content_type')):
            model = ctype.model_class()
            result[model] = {}
            for field, in_grp in groupby(out_grp, key=attrgetter('field')):
                result[model][field] = [
                    (i.text, i.object_id)
                    for i in in_grp
                ]

        return result


class UniqueWordQuerySet(QuerySet):
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

    def suggest_word_always(self, word, check_length=1, limit=20):
        """
        Suggest spelling for a single word

        This method always returns variations, even if word is a perfect match.
        """
        qs = self.annotate(dist=TrigramDistance('word', word))

        if isinstance(check_length, int):
            if check_length >= 0:
                qs = qs.annotate(length=Length('word')) \
                    .filter(
                        length__lte=len(word) + check_length,
                        length__gte=len(word) - check_length,
                    )
        elif check_length:
            raise ValueError('expect an int or evaluate to False')

        qs = qs.order_by('dist')

        if limit:
            return qs[:limit]
        else:
            return qs

    def suggest_phrase(self, txt, check_length=1, limit=20):
        """
        Suggest spelling for a phrase

        Returns a dict mapping words to list of closest matches.
        """
        # TODO: s/split/???/
        if isinstance(txt, str):
            txt = txt.split()
        suggestions = {word: [] for word in txt}
        for word in txt:
            matches = list(self.suggest_word(
                word,
                always=False,
                check_length=check_length,
                limit=limit,
            ))
            if not matches:
                return []
            if matches[0].word == word:
                # spelled correctly
                pass
            else:
                suggestions[word] = [(i.dist, i.word) for i in matches]

        return suggestions

    def suggest(self, txt, check_length=1, limit=20):
        """
        Get spelling suggestions

        This is the main entry-point for the whole spelling suggestion feature,
        usually called on the unfiltered table, as in
        UniqueWord.objects.all().suggest(...).
        """
        if isinstance(txt, str):
            txt = txt.split()
        suggestions = self.suggest_phrase(
            txt,
            check_length=check_length,
            limit=limit,
        )
        unique_dists = sorted(set((
            i
            for _, lst in suggestions.items()
            for i, _ in lst
        )))
        if not unique_dists:
            # all spelled correctly
            return suggestions

        # limit to most-similar matches
        limited = [(i, []) for i, _ in suggestions.items()]
        total = 0
        while total < limit:
            if unique_dists:
                cur = unique_dists.pop(0)
            for (_, lim_lst), (_, full_lst) in zip(limited, suggestions.items()):  # noqa:E501
                if full_lst and full_lst[0][0] <= cur:
                    lim_lst.append(full_lst.pop(0)[1])
                    total += 1

        return dict(limited)
