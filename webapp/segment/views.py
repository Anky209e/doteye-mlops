from django.shortcuts import render
from .models import ImageModel
from .forms import ImageForm
from seg_classes.segment import segment_image

def home(request):
    if request.method =="POST":        
        form = ImageForm(request.POST, request.FILES)
        try:
            if form.is_valid():
                form.save()
        except:
            pass

        img_path ="./media/uploaded_images/" + str(request.FILES["image_field"]).replace(" ", "_")

        print(img_path)
        result = segment_image(img_path)

        # result = None
        return render(request,"home.html",{"image_form": None, "list":result})
    else:
        image_form = ImageForm()

        return render(request, "home.html", {"image_form": image_form})

