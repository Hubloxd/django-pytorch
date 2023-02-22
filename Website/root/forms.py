from django import forms


class UploadForm(forms.Form):
    img = forms.FileField(widget=forms.FileInput(attrs={
        'class': 'input_class'
    }), label='Input label')
