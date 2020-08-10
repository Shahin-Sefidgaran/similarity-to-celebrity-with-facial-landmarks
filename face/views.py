from django.shortcuts import render
from django.db.models import F
from rest_framework import generics
import os
import time
import json
import threading

from .models import Faces
from .serializers import SongsSerializer
from facerecogntion.face_recognition_demo import main, Visualizer
from django.views.decorators.csrf import csrf_exempt

from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

# Create your views here.

visualizer = Visualizer()

class requestThread(threading.Thread):
    def __init__(self, threadID, name, path, train):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.path = path
        self.train = train
        self.res = ""

    def run(self):
        self.res = main(visualizer, self.path, self.train, self.name)

    def get_res(self):
        return self.res

counter = 0

class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        print("Here")
        global counter
        counter += 1
        print(request.FILES['photo'])
        uploaded_face = Faces(name=request.POST['name'], photo = request.FILES['photo'], model=request.POST['model'])
        uploaded_face.save()
        print(uploaded_face.photo)
        print(uploaded_face.name)
        s = str(uploaded_face.photo)
        s = s.split('/')

        ori_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(ori_path, 'media', 'media', s[1])

        start_time = time.time()
        th = requestThread(counter, uploaded_face.name, path, False)
        th.start()
        th.join()
        d = th.get_res()
        uploaded_face.result_photo = ""

        result_list = []
        result_dict = {}

        print("\n\n Time: ", str(time.time() - start_time), "s\n\n")
        print(d)
        for i in range(len(d)):
            if d[i] == "Unknown":
                continue
            else:
                s = d[i].split(' ')
                d[i] = "http://88.198.21.151:80/media/celebrities/" + s[0] + ".jpg"
                uploaded_face.result_photo = d[i]
                uploaded_face.similarity = s[1]
                uploaded_face.actors_name = s[0]

                result_dict = {"path": uploaded_face.result_photo, "similarity": uploaded_face.similarity, "name": uploaded_face.actors_name}
                result_list.append(result_dict)

        file_serializer = SongsSerializer(data={"name":uploaded_face.name, "photo":uploaded_face.photo, "result_photo":json.dumps(result_list)})
        if file_serializer.is_valid():
            file_serializer.save()

            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class FileTrainView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        uploaded_face = Faces(name=request.POST['name'], photo = request.FILES['photo'])
        uploaded_face.save()

        s = str(uploaded_face.photo)
        s = s.split('/')
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(path, 'media', 'media', s[1])
        d = str(main(visualizer, path, True, uploaded_face.name)).split('/')
        uploaded_face.result_photo = str(d)

        file_serializer = SongsSerializer(data={'name':uploaded_face.name, 'photo':uploaded_face.photo, 'result_photo':uploaded_face.result_photo})
        if file_serializer.is_valid():
            file_serializer.save()

            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ListFacesView(generics.ListAPIView):
    """
    Provides a get method handler.
    """
    queryset = Faces.objects.all()
    serializer_class = SongsSerializer

def index(request):
    return render(request, 'face/index.html')

@csrf_exempt
def send(request):
    uploaded_face = Faces(name=request.POST['name'], photo = request.FILES['myfile'])
    
    uploaded_face.save()
    #main(uploaded_face.photo)
    #args = build_argparser().parse_args()
    #visualizer = Visualizer(args)
    s = str(uploaded_face.photo)
    s = s.split('/')
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'media', 'media', s[1])
    d = str(main(path, False, uploaded_face.name)).split('/')
    uploaded_face.result_photo = "media/" + d[3]
    
    uploaded_face.save()
    #visualizer.run(args, path)
    
    return render(request, 'face/results.html', {'face': uploaded_face})
@csrf_exempt
def train(request):
    uploaded_face = Faces(name=request.POST['name'], photo = request.FILES['myfile'])
    
    uploaded_face.save()
    #main(uploaded_face.photo)
    #args = build_argparser().parse_args()
    #visualizer = Visualizer(args)
    s = str(uploaded_face.photo)
    s = s.split('/')
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(path, 'media', 'media', s[1])
    d = str(main(path, True, uploaded_face.name)).split('/')
    print(d)
    uploaded_face.result_photo = "media/" + d[3]
    
    uploaded_face.save()
    #visualizer.run(args, path)
    
    return render(request, 'face/results.html', {'face': uploaded_face})