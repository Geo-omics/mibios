from django import forms
from django.core.exceptions import ValidationError

from .templatetags.glamr_extras import human_lookups


class SearchForm(forms.Form):
    query = forms.CharField(
        strip=True,
        required=True,
        label='',
        # help_text='help text',
        initial='keyword search',
    )


class AdvancedSearchForm(SearchForm):
    search_all = forms.BooleanField(
        initial=False,
        required=False,
        # label=??,
        # help_text='Also show results for things not found in samples',
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


class QBuilderAddForm(QBuilderForm):
    """ Form to add a key/value filter item """
    key = forms.ChoiceField()
    lookup = forms.ChoiceField(choices=human_lookups.items())
    value = forms.CharField()

    def __init__(self, model, *args, path=[], **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        if path:
            self.fields['path'].initial = ','.join([str(i) for i in path])
        else:
            # Empty path means root node, but can't set field value to '' here
            # because on the return (e.g. the POST request) this would be
            # interpreted as missing field, but we have path as 'required'
            # so we chose 'None' as the special value
            self.fields['path'].initial = 'None'
        self.set_key_choices()

    def set_key_choices(self):
        lst = []
        for i in self.model.get_related_accessors():
            lst.append((i, i.replace('__', ' -> ')))
        self.fields['key'].choices = lst