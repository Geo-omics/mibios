from mibios.query import QuerySet


class TaxNodeQuerySet(QuerySet):
    def str_only(self):
        """ Defer fields not needed to run __str__() on instances """
        return self.only('name', 'rank')
