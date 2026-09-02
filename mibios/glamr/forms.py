from django import forms
from django.core.exceptions import ValidationError
from django.forms.widgets import HiddenInput, Select, TextInput
# from django.utils.safestring import mark_safe

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, ButtonHolder, Submit

from .models import Dataset
from .search_fields import search_fields
from .templatetags.glamr_extras import human_lookups

from mibios.umrad.models import UniRef50, UniRef90, UniRef100


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


class UniRefIDForm(forms.Form):
    uniref_id = forms.CharField(
        label='UniRef ID',
        strip=True,
        required=True,
        help_text='UniRef50/90/100 ID / accession with or without prefix',
    )

    @classmethod
    def from_data(cls, data):
        """
        Instantiate form depending on data in request.GET

        Returns bound form if the data indicates that the request is via form
        submission, otherwise returns an unbound form.
        """
        if all(i in data for i in cls.base_fields):
            return cls(data)
        else:
            return cls()

    def clean_uniref_id(self):
        """
        Sets the record attribute with the model instance if one if found

        If no record exists, then an error is added to the field.
        """
        uniref_id = self.cleaned_data['uniref_id']
        if obj := self.get_object(uniref_id):
            self.record = obj
        else:
            self.add_error(
                'uniref_id',
                f'We do not have a UniRef record with ID {uniref_id}'
            )
        return uniref_id

    @classmethod
    def get_object(cls, accession):
        """
        Get matching UniRef record

        If the submitted accession has one of the usual prefixes, e.g.
        UniRef100_ then only the corresponding model is tested.  Otherwise, if
        the accession has no prefix, then the most specific UniRef100 models is
        tested first and the least specific clustering (UniRef50) is tested
        last.  The first hit is returned.  If no match is found then None is
        returned.
        """
        models = (UniRef100, UniRef90, UniRef50)
        for model in models:
            prefix = model._meta.model_name.casefold() + '_'
            if accession.casefold().startswith(prefix):
                accession = accession[len(prefix):]
                candidate_models = (model,)
                break
        else:
            candidate_models = models

        for model in candidate_models:
            try:
                return model.objects.get(accession=accession)
            except model.DoesNotExist:
                continue
        return None
