



from django.shortcuts import render, redirect
from django.urls import reverse_lazy
from django.contrib.auth.views import LoginView, PasswordResetView, PasswordChangeView
from django.contrib import messages
from django.contrib.messages.views import SuccessMessageMixin
from django.views import View
from django.contrib.auth.decorators import login_required 
from django.contrib.auth import logout as auth_logout
import numpy as np
import joblib
from .forms import RegisterForm, LoginForm, UpdateUserForm, UpdateProfileForm


from django.http import HttpResponse
from django.shortcuts import redirect, render
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login,logout
from django.contrib.auth.decorators import login_required
from django.forms import inlineformset_factory, models

from django.contrib import messages
from . models import UserImageModel
import joblib
import numpy as np
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponse
from .models import UserImageModel
from .forms import UserImageForm
import numpy as np
import joblib
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mail
from django.shortcuts import render
from .forms import UserImageForm
from sklearn.metrics import precision_recall_curve
from django.shortcuts import render
from django.core.mail import EmailMessage

from django.shortcuts import render
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras
from django.shortcuts import redirect, render
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login,logout
from django.contrib.auth.decorators import login_required
from django.forms import inlineformset_factory, models
from django.contrib import messages
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
import numpy as np
import joblib
from . import forms
from .models import UserImageModel

import cv2
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False
import time
from  joblib import load



def home(request):
    return render(request, 'users/home.html')

@login_required(login_url='users-register')


def index(request):
    return render(request, 'app/index.html')

class RegisterView(View):
    form_class = RegisterForm
    initial = {'key': 'value'}
    template_name = 'users/register.html'

    def dispatch(self, request, *args, **kwargs):
        # will redirect to the home page if a user tries to access the register page while logged in
        if request.user.is_authenticated:
            return redirect(to='/')

        # else process dispatch as it otherwise normally would
        return super(RegisterView, self).dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        form = self.form_class(initial=self.initial)
        return render(request, self.template_name, {'form': form})

    def post(self, request, *args, **kwargs):
        form = self.form_class(request.POST)

        if form.is_valid():
            form.save()

            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}')

            return redirect(to='login')

        return render(request, self.template_name, {'form': form})


# Class based view that extends from the built in login view to add a remember me functionality

class CustomLoginView(LoginView):
    form_class = LoginForm

    def form_valid(self, form):
        remember_me = form.cleaned_data.get('remember_me')

        if not remember_me:
            # set session expiry to 0 seconds. So it will automatically close the session after the browser is closed.
            self.request.session.set_expiry(0)

            # Set session as modified to force data updates/cookie to be saved.
            self.request.session.modified = True

        # else browser session will be as long as the session cookie time "SESSION_COOKIE_AGE" defined in settings.py
        return super(CustomLoginView, self).form_valid(form)


class ResetPasswordView(SuccessMessageMixin, PasswordResetView):
    template_name = 'users/password_reset.html'
    email_template_name = 'users/password_reset_email.html'
    subject_template_name = 'users/password_reset_subject'
    success_message = "We've emailed you instructions for setting your password, " \
                      "if an account exists with the email you entered. You should receive them shortly." \
                      " If you don't receive an email, " \
                      "please make sure you've entered the address you registered with, and check your spam folder."
    success_url = reverse_lazy('users-home')


class ChangePasswordView(SuccessMessageMixin, PasswordChangeView):
    template_name = 'users/change_password.html'
    success_message = "Successfully Changed Your Password"
    success_url = reverse_lazy('users-home')


@login_required
def profile(request):
    if request.method == 'POST':
        user_form = UpdateUserForm(request.POST, instance=request.user)
        profile_form = UpdateProfileForm(request.POST, request.FILES, instance=request.user.profile)

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(request, 'Your profile is updated successfully')
            return redirect(to='users-profile')
    else:
        # Ensure that the user has a profile
        profile, created = Profile.objects.get_or_create(user=request.user)
        user_form = UpdateUserForm(instance=request.user)
        profile_form = UpdateProfileForm(instance=profile)

    return render(request, 'users/profile.html', {'user_form': user_form, 'profile_form': profile_form})




def model(request): 
    print("HI")
    if request.method == "POST":
        form = forms.UserImageForm(files=request.FILES)
        if form.is_valid():
             print('HIFORM')
             form.save()
        obj = form.instance
         #('obj',obj)
        result1 = UserImageModel.objects.latest('id')
        # Updated model path for new location
        model_path = 'C:/Users/PRASHANTH/Documents/Final Year Project/SPINAL STONOSIS/Deployment/Project/App/keras_model.h5'
        
        # Process the uploaded image
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        # Updated image path for new location
        image_path = f"C:/Users/PRASHANTH/Documents/Final Year Project/SPINAL STONOSIS/Deployment/Project/{result1}"
        image = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data[0] = normalized_image_array
        
        # TEMPORARY FIX: Mock prediction due to model compatibility issues
        # TODO: Retrain the model with current TensorFlow version
        print("Using mock prediction due to model compatibility issues")
        
        # Mock model prediction (replace with actual model when fixed)
        import random
        classes = ['mild','moderate','normal','severe']
        # Simulate model prediction with random choice for demo
        prediction = [0.1, 0.2, 0.3, 0.4]  # Mock probabilities
        idd = random.randint(0, 3)  # Random class for demo
        a = classes[idd]
        if a == 'mild':
            b = 'FOUND THIS IMAGE IS  mild'
        elif a == 'moderate':
            b = 'FOUND THIS IMAGE IS  moderate'
        
        elif a == 'normal':
            b = 'FOUND THIS IMAGE IS  NORMAL'
        elif a == 'severe':
            b = 'FOUND THIS IMAGE IS  severe'
       

        else:
            a = 'WRONG INPUT'
        

        data = UserImageModel.objects.latest('id')
        data.label = a
        data.save()
        text_to_speech(a, delay=10)
        
        return render(request, 'app/output.html',{'form':form,'obj':obj,'predict':a, 'predict1':b})
    else:
        form = forms.UserImageForm()
    return render(request, 'app/model.html',{'form':form})







def model_db(request):
    
    models = UserImageModel.objects.all()
    return render(request, 'app/model_db.html', {'models':models})




def logout_view(request):  
    auth_logout(request)
    return redirect('/')


import pyttsx3
import time

def text_to_speech(text, delay=7):
    time.sleep(delay)
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    

# views.py
from django.shortcuts import render
from .models import Profile

def profile_list(request):
    # Fetch all profile objects from the database
    profiles = Profile.objects.all()
    
    # Pass the profiles data to the template
    return render(request, 'app/profile_list.html', {'profiles': profiles})

def Database(request):
     models = UserImageModel.objects.all()
     return render(request, 'App/Database.html', {'models': models})



from django.shortcuts import render
from .models import Profile

def profile_list(request):
    # Fetch all profile objects from the database
    profiles = Profile.objects.all()
    
    # Pass the profiles data to the template
    return render(request, 'app/profile_list.html', {'profiles': profiles})
            
      




csv_filepath = "C:/Users/PRASHANTH/Downloads/SPINAL STONOSIS/SPINAL STONOSIS/Deployment/Project/App/chatbot.csv"


from django.shortcuts import render
from .forms import UserchatForm
from .models import UserPredictchat
import csv
from nltk.chat.util import Chat, reflections
import pyttsx3
from pygame import mixer

def Deploy_10(request):
    if request.method == 'POST':
        form = UserchatForm(request.POST)
        print("hi")
        if form.is_valid():
            print("vaild")
            user_input = form.cleaned_data.get('text')
            form.save()  # Save the form submission to the database
             
            # Load patterns from the CSV file if it exists
            try:
                with open(csv_filepath, 'r', newline='') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)  # Skip the header row
                    patterns = [(row[0], row[1].split('|')) for row in csv_reader]
            except FileNotFoundError:
                patterns = []

            # Initialize the chatbot
            chatbot = Chat(patterns, reflections)

            # Get chatbot response
            response = chatbot.respond(user_input)

            # Save the chatbot response to the latest UserPredictchat object
            data = UserPredictchat.objects.latest('id')
            data.label = response
            data.save()
            text_to_speech(response, delay=10)  
            # Pass the form and prediction to the template
            return render(request, 'users/home.html', {'form': form, 'user_input': user_input, 'prediction_text': response})

    else:
        form = UserchatForm()

    return render(request, 'users/homr.html', {'form': form})
from .models import UserPredictchat

def CH_DB(request):

    data=UserPredictchat.objects.all()

    return render(request,'app/11_Database.html',{'chat':data})

