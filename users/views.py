from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
from django.conf import settings

import seaborn as sns
from django.core.files.storage import FileSystemStorage

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid=request.POST.get("loginid")
        password=request.POST.get("pswd")
        print(loginid)
        print(password)
        try:
            check=UserRegistrationModel.objects.get(loginid=loginid,password=password)
            status=check.status
            if status=="activated":
                request.session['id']=check.id
                request.session['loginid']=check.loginid
                request.session['password']=check.password
                request.session['email']=check.email
                return render(request,'users/UserHome.html',{})
            else:
                messages.success(request,"your account not activated")
            return render(request,"UserLogin.html")
        except Exception as e:
            print('=======>',e)
        messages.success(request,'invalid details')
    return render(request,'UserLogin.html',{})
    
def UserHome(request):
    return render(request,"users/UserHome.html",{})



  
def view_data(request):
    from django.conf import settings
    import pandas as pd


    path= settings.MEDIA_ROOT + "//" + 'disease_dataset.csv'
    d = pd.read_csv(path)
    

    context = {'dataset': d}
    return render(request,'users/dataset.html',context) 
#import neccesary model for model training
from django.conf import settings
from django.shortcuts import render
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB  # for binary data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Lad data from CSV files
# Load data from CSV files
path1 = settings.MEDIA_ROOT + "//" + 'dataset.csv'
path2 = settings.MEDIA_ROOT + "//" + 'Symptom-severity.csv'
df = pd.read_csv(path1)
path1 = settings.MEDIA_ROOT + "//" + 'dataset.csv'
df2 = pd.read_csv(path2)
# Create a dictionary from df2
dict_ = df2.to_dict()
appl = dict(zip(dict_["Symptom"].values(), dict_["weight"].values()))
# Preprocess the data
df.fillna(0, inplace=True)
# Define a function to map symptoms to weights
def fun(x):
    try:
        return appl.get(x.strip(), 0)
    except:
        return 0
df_conv = df.drop(columns=["Disease"]).applymap(fun)
df_new = df_conv.join(df["Disease"])
# Create a mapping of disease labels
key = [i for i in range(1, len(df['Disease'].unique()) + 1)]
values = np.unique(df["Disease"])
dict_2 = {k: v for k, v in zip(key, values)}
# Encode the target variable
x = df_new[['Symptom_1','Symptom_2','Symptom_3','Symptom_4','Symptom_5','Symptom_6','Symptom_7','Symptom_8','Symptom_9','Symptom_10']]
y = LabelEncoder().fit_transform(df_new.iloc[:, -1])  # output(target variable) = diseases
# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
def training(request):
    # Train Decision Tree Classifier
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    dt_results = train_model(request, dt, x_test, y_test, values, "Decision Tree")

    # Train Random Forest Classifier
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    rf_results = train_model(request, rf, x_test, y_test, values, "Random Forest")
    
    #train Naive Bayes
    naive= BernoulliNB()
    naive.fit(x_train, y_train)
    nb_results=train_model(request, naive, x_test, y_test, values, "Naive Bayes")
    
    # Train Support Vector Classifier
    svc = SVC()
    svc.fit(x_train, y_train)
    svc_results = train_model(request, svc, x_test, y_test, values, "Support Vector Classifier")

    #Train KNeighborsClassifier
    knc=KNeighborsClassifier()
    knc.fit(x_train, y_train)
    knc_results=train_model(request, knc, x_test, y_test, values, "KNeighborsClassifier")

    #Train Logistic Regressiion
    lr =LogisticRegression()
    lr.fit(x_train,y_train)
    lr_results=train_model(request, lr, x_test, y_test, values, "LogisticRegression")



    results_list= [dt_results,rf_results,nb_results,svc_results,knc_results,lr_results]

    return render(request, "users/modelresults.html",{'results_list':results_list})

def train_model(request, model, x_test, y_test, values, model_name):
    # Make predictions using the trained model
    model_pred = model.predict(x_test)

    # Calculate accuracy and other metrics
    model_acc = accuracy_score(y_test, model_pred)
    model_precision = precision_score(y_test, model_pred, average='macro')
    model_recall = recall_score(y_test, model_pred, average='macro')
    model_f1 = f1_score(y_test, model_pred, average='macro')
    model_cm = confusion_matrix(y_test, model_pred)

    # Display the confusion matrix
    cm = ConfusionMatrixDisplay(confusion_matrix=model_cm, display_labels=values)
    cm.plot()
    plt.xticks(rotation=90)
    plt.title(f'{model_name} Confusion Matrix')

    # Render the results in an HTML template
    return {
        'model_name': model_name,
        'model_acc': model_acc,
        'model_precision': model_precision,
        'model_recall': model_recall,
        'model_f1': model_f1,
    }

def random_forest(request):
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    # Make predictions using the trained model
    return rf


def prediction(request):
    if request.method == 'POST':
        try:
            # Get user input values from the form
            Symptom_1 = request.POST['Symptom_1']
            Symptom_2 = request.POST['Symptom_2']
            Symptom_3 = request.POST['Symptom_3']
            Symptom_4 = request.POST['Symptom_4']
            Symptom_5 = request.POST['Symptom_5']
            Symptom_6 = request.POST['Symptom_6']
            Symptom_7 = request.POST['Symptom_7']
            Symptom_8 = request.POST['Symptom_8']
            Symptom_9 = request.POST['Symptom_9']
            Symptom_10 = request.POST['Symptom_10']

            # Create a DataFrame with the user's input data
            input_str = pd.DataFrame({
                'Symptom_1': [Symptom_1],
                'Symptom_2': [Symptom_2],
                'Symptom_3': [Symptom_3],
                'Symptom_4': [Symptom_4],
                'Symptom_5': [Symptom_5],
                'Symptom_6': [Symptom_6],
                'Symptom_7': [Symptom_7],
                'Symptom_8': [Symptom_8],
                'Symptom_9': [Symptom_9],
                'Symptom_10': [Symptom_10],
            })
            input_str.fillna(0, inplace=True)

            # Define a function to map symptoms to weights
            def fun(x):
                try:
                    return appl.get(x.strip(), 0)
                except:
                    return 0

            input_data = input_str.applymap(fun)

            rf_classifier = random_forest(request)

            # Predict disease for the user's input
            predicted_id = int(rf_classifier.predict(input_data))
            predicted_disease = dict_2[predicted_id]

            return render(request, "users/result.html", {"predicted_disease": predicted_disease})

        except Exception as e:
            # Handle errors and exceptions
            return render(request, "users/prediction.html", {"error_message": str(e)})

    else:
        return render(request, 'users/prediction.html')

        





