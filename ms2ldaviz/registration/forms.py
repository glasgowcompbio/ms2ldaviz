from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import Profile


class UserForm(UserCreationForm):
    first_name = forms.CharField(max_length=100) # Required
    last_name = forms.CharField(max_length=100)  # Required
    email = forms.EmailField(max_length=100)     # Required
    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(
            attrs={'class': 'form-control', 'style': 'width: 300px'}),
    )
    password2 = forms.CharField(
        label="Confirm password",
        widget=forms.PasswordInput(
            attrs={'class': 'form-control', 'style': 'width: 300px'}),
    )

    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email']
        labels = {
                     'username': 'Username',
                     'first_name': 'First Name',
                     'last_name': 'Last Name',
                     'email': 'Email Address',
                 },
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'style': 'width: 300px'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control', 'style': 'width: 300px'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control', 'style': 'width: 300px'}),
            'email': forms.EmailInput(attrs={'class': 'form-control', 'style': 'width: 300px'}),
        }


class ProfileForm(forms.ModelForm):
    class Meta:
        model = Profile
        fields = ['affiliation', 'country']
        labels = {
                     'affiliation': 'Organisation that you are affiliated to',
                     'country': 'Country'
                 },
        widgets = {
            'affiliation': forms.TextInput(attrs={'class': 'form-control', 'style': 'width: 300px'}),
            'country': forms.Select(attrs={'class': 'form-control', 'style': 'width: 300px'}),
        }
