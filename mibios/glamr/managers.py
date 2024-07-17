from django.db.models import Manager


class dbstatManager(Manager):
    def get_queryset(self):
        # see: https://www.sqlite.org/dbstat.html
        # use extra+where instead of filter() since aggregate is not a field
        return super().get_queryset().extra(where=['aggregate = True'])
