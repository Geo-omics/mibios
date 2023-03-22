from django.contrib.postgres.search import SearchVectorField
from django.db.migrations import AddField
from django.db.models import NOT_PROVIDED


class AddTSVectorField(AddField):
    """
    Operation that adds a postgresql tsvector column

    The normal Addfield works just fine with SearchVectorField but we want to
    go a bit beyond:

    1. let the tsvector column be kept up-to-date automatically w/o triggers
    2. degrade gracefully when using the sqlite backend

    How to use this:

    1. Add a SearchVectorField field to a model
    2. Run ./manage.py makemigrations as usual
    3. Edit the migration file:
        - if the op is a CreateModel then first remove the line for the search
          vector field and instead add a corresponding AddField operation
        - change class of operation item from AddField to AddTSVectorField
        - add a keyword parameter target_field_name=<name of text column>

    Certain field options (null, unique) don't make much sense in a tsvector
    column and are ignored in the creation of the column on postgresql DBs.

    For backends other than postgresql and the reverse direction we fall-back
    to the normal AddField operation.
    """

    def __init__(self, model_name, name, field, target_field_name=None,
                 **kwargs):
        if not isinstance(field, SearchVectorField):
            raise TypeError('field must be a SeachVectorField')

        if target_field_name is None:
            raise ValueError('target_field_name kwarg is required')

        self.target_field_name = target_field_name
        super().__init__(model_name, name, field, **kwargs)

    def describe(self):
        return f'{super().describe()} (tsvector column)'

    def database_forwards(self, app_label, schema_ed, from_st, to_st):
        if schema_ed.connection.vendor != 'postgresql':
            # the normal AddField seems to work fine for sqlite, it simply adds
            # a column of type "tsvector" (it unlikely to break anythying as
            # long as it's not used).  No idea what might happen on other DB
            # backends.
            super().database_forwards(app_label, schema_ed, from_st, to_st)
            return

        # For postgres start a bit like AddField.database_forward but then we
        # run our own sql and not schema_editor.add_field()
        to_model = to_st.apps.get_model(app_label, self.model_name)
        if not self.allow_migrate_model(schema_ed.connection.alias, to_model):  # noqa: E501
            return

        from_model = from_st.apps.get_model(app_label, self.model_name)
        field = to_model._meta.get_field(self.name)
        if not self.preserve_default:
            field.default = self.field.default

        table = from_model._meta.db_table
        target = self.target_field_name

        sql = f"ALTER TABLE {table} ADD COLUMN {field.column} tsvector " \
              f"GENERATED ALWAYS AS (to_tsvector('english', {target})) STORED"
        schema_ed.execute(sql)

        if not self.preserve_default:
            field.default = NOT_PROVIDED
