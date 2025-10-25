from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
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
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def DatasetView(request):
    return render(request, 'users/viewdataset.html', {})

def trainning(request):
    import numpy as np # linear algebra
    import pandas as pd  # data processing
    import os #  to interact with files using there paths
    from sklearn.datasets import load_files
    #The path of our data on drive
    path = settings.MEDIA_ROOT + '\\' + 'flowers'
    
    
    #Loading our Data
    data = load_files(path)
    folders=os.listdir(path)
    print(folders)
    X = np.array(data['filenames'])
    y = np.array(data['target'])
    labels = np.array(data['target_names'])
    
    # How the arrays look like?
    print('Data files - ',X)
    print('Target labels - ',y)
    pyc_file_pos = (np.where(file==X) for file in X if file.endswith(('.pyc','.py')))
    for pos in pyc_file_pos:
        X = np.delete(X,pos)
        y = np.delete(y,pos)
        
    print('Number of training files : ', X.shape[0])
    print('Number of training targets : ', y.shape[0])
    #from keras.preprocessing.image import img_to_array, load_img
    from tensorflow.keras.utils import img_to_array, load_img

    def convert_img_to_arr(file_path_list):
        arr = []
        #size=64,64
        img_width, img_height = 150,150
        for file_path in file_path_list:
            img = load_img(file_path, target_size = (img_width, img_height))
            img = img_to_array(img)
            arr.append(img)
            #arr.append(cv2.resize(img,size))
        return arr
    
    X = np.array(convert_img_to_arr(X))
    print(X.shape) 
    print('First training item : ',X[0])
    X = X.astype('float32')/255
 
    # Let's confirm the number of classes :) 
    no_of_classes = len(np.unique(y))
    no_of_classes
    from keras.utils import np_utils

# let's converts a class vector (integers) to binary class matrix.
    y = np.array(np_utils.to_categorical(y,no_of_classes))
    y[0]
    from sklearn.model_selection import train_test_split

# let's splite the data into subsets and explore their shapes !

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    print('The test Data Shape ', X_test.shape[0])
    
    X_test, X_valid, y_test, y_valid = train_test_split(X_test,y_test, test_size = 0.5)
    print('The training Data Shape ', X_valid.shape[0])
    import keras
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ReduceLROnPlateau
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=X_train.shape[1:], activation='relu', name='Conv2D_1'))
    model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', name='Conv2D_2'))
    model.add(MaxPool2D(pool_size=(2,2), name='Maxpool_1'))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_3'))
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', name='Conv2D_4'))
    model.add(MaxPool2D(pool_size=(2,2), name='Maxpool_2'))
    model.add(Dropout(0.25))
        
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='relu', name='Conv2D_5'))
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', name='Conv2D_6'))
    model.add(MaxPool2D(pool_size=(2,2), name='Maxpool_3'))

    model.add(Flatten())
    model.add(Dense(units=512, activation='relu', name='Dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(units=128, activation='relu', name='Dense_2'))
    model.add(Dense(units=no_of_classes, activation='softmax', name='Output'))
    model.summary()

    from keras.optimizers import RMSprop

    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    import time
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    
    # Time to train our model !
    epochs = 1
    batch_size=32
    
    train_datagen = ImageDataGenerator(
            rotation_range=10,  
            zoom_range = 0.1, 
            width_shift_range=0.1,
            height_shift_range=0.1,  
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator()
    
    train_generator = train_datagen.flow(
        X_train,y_train,
        batch_size=batch_size)
    
    validation_generator = test_datagen.flow(
        X_valid,y_valid,
        batch_size=batch_size)
    
    checkpointer = ModelCheckpoint(filepath = "/gdrive/My Drive/PId_Best.h5", save_best_only = True, verbose = 1)
    learning_rate_reduction=ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose = 1, factor = 0.5, minlr = 0.00001)
    

    start = time.time()
    
    # let's get started !
    
    history=model.fit_generator(train_generator,
                                epochs=epochs,
                                validation_data = validation_generator,
                                verbose=1,
                                steps_per_epoch=len(X_train) // batch_size,
                                #validation_steps=len(X_valid) //batch_size,
                                callbacks=[checkpointer, learning_rate_reduction])
    
    end = time.time()
    model.save("model.h5")
    
    duration = end - start
    print ('\n This Model took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, epochs) )
    (eval_loss, eval_accuracy) = model.evaluate(  
     X_test, y_test, batch_size=batch_size, verbose=2)
 
    print("Accuracy: {:.2f}%".format(eval_accuracy * 100))  
    print("Loss: {}".format(eval_loss))
    # print("Accuracy":"eval_accuracy * 100")  
    # print("Loss":"eval_loss")

    #return render(request,"users/training.html",{"Accuracy: {:.2f}%".format(eval_accuracy * 100),"Loss: {}".format(eval_loss)})
    return render(request,"users/training.html",{"Accuracy":eval_accuracy * 100,"Loss":eval_loss})






def prediction(request):
    if request.method=='POST':
        from django.core.files.storage import FileSystemStorage
        image_file = request.FILES['file']
        fs = FileSystemStorage(location="media/rice_test/")
        filename = fs.save(image_file.name, image_file)
        # detect_filename = fs.save(image_file.name, image_file)
        uploaded_file_url = "/media/daisy/" + filename  # fs.url(filename)
        file = settings.MEDIA_ROOT + '\\' + 'rice_test' + '\\' + filename
        print("Image path ", uploaded_file_url)
        from .utility.predictions import prediction
        result = prediction(file)
        print("Result=", result)
        return render(request, "users/testform.html", {'path': uploaded_file_url,'result': result})
    else:
        return render(request, "users/testform.html",{})
    



