from django import forms
from django.core.exceptions import ValidationError
from django.forms.widgets import HiddenInput, Select, TextInput
# from django.utils.safestring import mark_safe

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, ButtonHolder, Submit

from mibios import QUERY_FORMAT
from .models import Dataset
from .search_fields import search_fields
from .templatetags.glamr_extras import human_lookups


class DatasetFilterFormHelper(FormHelper):
    model = Dataset
    # form_tag = False
    # Adding a Filter Button
    layout = Layout('name', ButtonHolder(
        Submit('submit', 'Filter', css_class='button white right')
    ))


class RenderMixin:
    """ extra (experimental) rendering methods for forms """
    def as_bootstrap(self):
        "Return this form rendered as HTML for bootstrap 5 'single line'."
        html = self._html_output(
            normal_row='%(errors)s<span class="input-group-text">%(label)s</span>%(field)s%(help_text)s<span></span>',  # noqa: E501
            error_row='</div><div class="input-group input-group-sm mb-3 alert-danger">%s</div><div class="input-group">',  # noqa: E501
            row_ender='</span>',  # noqa:E501 ## hack with the span thing at the normal row end to make hidden fields work
            help_text_html='<span class="input-group-text">%s</span>',
            errors_on_separate_row=False,
        )
        # FIXME: ugly hack (that doesn't work)
        # html = mark_safe(html.replace('<label', '<label class="form-label"'))
        return html


class SearchForm(forms.Form):
    query = forms.CharField(
        strip=True,
        required=True,
        label='',
        # help_text='help text',
        initial='keyword search',
    )
    limit = forms.IntegerField(
        required=False,
        widget=HiddenInput,
    )


class AdvancedSearchForm(SearchForm):
    MODEL_CHOICES = [('', '')] + [
        (i._meta.model_name, i._meta.model_name.upper())
        for i in search_fields.keys()
    ]
    model = forms.ChoiceField(choices=MODEL_CHOICES, required=False)
    field_data_only = forms.BooleanField(
        initial=True,
        required=False,
        # label=??,
        # help_text='Restrict results to things found in field samples',
    )


class QBuilderForm(forms.Form):
    """ Form to manipulate a Q object """
    path = forms.CharField(widget=forms.HiddenInput())

    def clean_path(self):
        value = self.cleaned_data['path']
        if value == 'None':
            # 'None' encodes an empty list here.
            return []
        try:
            return [int(i) for i in value.split(',')]
        except Exception as e:
            raise ValidationError(
                f'failed parsing path field content: {e} {value=}'
            )


class QLeafEditForm(RenderMixin, QBuilderForm):
    """ Form to add/edit a key/value filter item """
    key = forms.ChoiceField(widget=Select(attrs={'class': 'form-select'}))
    lookup = forms.ChoiceField(choices=human_lookups.items(),
                               widget=Select(attrs={'class': 'form-select'}))
    value = forms.CharField(widget=TextInput(attrs={'class': 'form-control'}))
    add_mode = forms.BooleanField(
        required=False,  # to allow this to be False
        widget=forms.HiddenInput()
    )

    def __init__(self, model, *args, add_mode=True, path=[], key=None,
                 lookup=None, value=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.fields['add_mode'].initial = add_mode

        if path:
            self.fields['path'].initial = ','.join([str(i) for i in path])
        else:
            # Empty path means root node, but can't set field value to '' here
            # because on the return (e.g. the POST request) this would be
            # interpreted as missing field, but we have path as 'required'
            # so we chose 'None' as the special value
            self.fields['path'].initial = 'None'

        if key is not None:
            self.fields['key'].initial = key
        if lookup is not None:
            self.fields['lookup'].initial = lookup
        if value is not None:
            self.fields['value'].initial = value

        self.set_key_choices()

    def set_key_choices(self):
        lst = []
        for path in self.model.get_related_fields():
            accessor = '__'.join([i.name for i in path])
            names = [getattr(i, 'verbose_name', i.name) for i in path]
            humanized = ' -> '.join(names)
            lst.append((accessor, humanized))
        self.fields['key'].choices = lst


class ExportFormatForm(forms.Form):
    """
    Form to pick download file formatting options

    For use with glamr.views.ExportMixin.  This is similar in purpose but
    evolved from mibios.forms.ExportFormatForm.
    """
    export_format = forms.ChoiceField(
        widget=forms.RadioSelect(attrs={'class': None}),
        # choices/initial set by constructor
        label='file format',
        required=False,
    )
    export_deflate = forms.ChoiceField(
        widget=forms.RadioSelect(attrs={'class': None}),
        # choices/initial set by constructor
        label='compression format',
        required=False,
    )

    @classmethod
    def factory(cls, view, name=None, base=None, opts=None):
        """
        Return a form class to work with given ExportBaseMixin derived view.

        This method can be called as super().factory() from a deriving class
        and then will just add the parent attributes.
        """
        base = base or (cls, )
        if opts is None:
            opts = {}

        # view.FORMATS is list of triplets (format, file suffix, renderer
        # class) where format is a simple format string or format string and
        # compression format separated by a slash.
        fmt_choices = set()
        defl_choices = set()
        for fmt_code, _, _ in view.FORMATS:
            file_code, _, defl_code = fmt_code.partition('/')
            fmt_choices.add(file_code)
            if defl_code:
                defl_choices.add(defl_code)

        default_fmt, _, default_defl = view.DEFAULT_FORMAT.partition('/')
        fmt_opts = view.get_format_choices()
        choices = fmt_opts['choices']
        defaults = fmt_opts['defaults']

        if len(choices['format']) > 1:
            opts['format_choices'] = choices['format']
            opts['default_format'] = defaults['format']
        else:
            opts['export_format'] = None

        if len(choices['deflate']) > 1:
            opts['deflate_choices'] = choices['deflate']
            opts['default_deflate'] = defaults['deflate']
        else:
            opts['export_deflate'] = None

        name = name or 'Auto' + cls.__name__
        return type(name, base, opts)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'export_format' in self.fields:
            self.fields['export_format'].choices = self.format_choices
            self.fields['export_format'].initial = self.default_format
        if 'export_deflate' in self.fields:
            self.fields['export_deflate'].choices = self.deflate_choices
            self.fields['export_deflate'].initial = self.default_deflate

    def add_prefix(self, field_name):
        """
        API abuse to correctly set the HTML input attribute
        """
        # TODO still needed?
        if field_name == 'format':
            field_name = QUERY_FORMAT
        return super().add_prefix(field_name)
