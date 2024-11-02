from flask_socketio import SocketIO
import torch
import cv2
import mediapipe as mp
from flask import Flask, render_template, Response
from flask import _app_ctx_stack as _request_ctx_stack
import os
import numpy as np
from ultralytics import YOLO


# Charger le modèle YOLOv5
model = YOLO('bestv8.pt')
model_person = torch.hub.load('ultralytics/yolov5:master', 'custom', path='yolov5s.pt')

save_dir = 'images_sans_epi'
os.makedirs(save_dir, exist_ok=True)
frame_count = 0
erreur_compt=0
messages=[]
# Initialisation des listes
personnes_sans_casques = []
personnes_sans_blouses = []
personnes_sans_soudages = []
personnes_sans_chaussures = []
personnes_sans_gloves = []
personnes_sans_goggles = []

personnes_sans_EPI=[]
# Initialiser l'objet Pose de Mediapipe
pose = mp.solutions.pose.Pose()

# Créer une instance Flask
app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="http://localhost:5000")
message = ''
compt =0
csq , bls , glv , chauss , gog = 0,0,0,0,0

frame_counter = 0
# Fonction pour lire le flux vidéo en temps réel
def generate_frames():
    global frame_counter,compt,erreur_compt,csq,bls,glv,chauss,gog
    video = cv2.VideoCapture('video_soudage.mp4')
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    
    # Ajuster la fréquence d'images du flux vidéo
    video.set(cv2.CAP_PROP_FPS, fps)
    while True:
        success, frame = video.read()
        if not success:
            break
        compt +=1
        if isinstance(frame, np.ndarray)  :
            
            # ici la detection des personnes et des objets dans la frame courante
            results = model.track(frame)[0]
            persons = model_person(frame).xyxy[0].numpy()[model_person(frame).xyxy[0].numpy()[:, -1] == 0]

            
            # detection des casques dans la frame courante
            casques = results.boxes.xyxy.numpy()[ np.where(results.boxes.cls.cpu().numpy() == 0.0)[0] ]
            class_indices = results.boxes.cls.cpu().tolist()
            print("Class indices:", class_indices)

            # detection des blouses dans la frame courante
            blouses = results.boxes.xyxy.numpy()[np.where(results.boxes.cls.cpu().numpy() == 2.0)[0]]
            print(len(blouses))

            # detection des chaussures
            chaussures = results.boxes.xyxy.numpy()[np.where(results.boxes.cls.cpu().numpy() == 3.0)[0]]

            # detection des gloves
            gloves = results.boxes.xyxy.numpy()[ np.where(results.boxes.cls.cpu().numpy() == 4.0)[0] ]

            # detection des goggles
            goggles = results.boxes.xyxy.numpy()[  np.where(results.boxes.cls.cpu().numpy() == 5.0)[0] ]
            soudages = results.boxes.xyxy.numpy()[ np.where(results.boxes.cls.cpu().numpy() == 1.0)[0] ]
            for person in persons:
                
                personnes_sans_casques = []
                personnes_sans_blouses = []
                personnes_sans_soudages = []
                personnes_sans_chaussures = []
                personnes_sans_gloves = []
                personnes_sans_goggles = []
                personnes_sans_epi=[]
                
                xp, yp, wp, hp = person[:4]
                if isinstance(xp, np.ndarray):
                    xp, yp, wp, hp = xp[0], yp[0], wp[0], hp[0]
                # prendre une copie de l'image ou se trouve la personne courante
                copie_person = frame[int(yp):int(hp), int(xp):int(wp)]
                height,width,_ =copie_person.shape

                copie_person = cv2.cvtColor(copie_person, cv2.COLOR_BGR2RGB)
                rslts = pose.process(copie_person)
                if rslts is not None and rslts.pose_landmarks is not None:
                    head = rslts.pose_landmarks.landmark[0]
                    body1 = rslts.pose_landmarks.landmark[23]
                    body2 = rslts.pose_landmarks.landmark[24]
                    left = rslts.pose_landmarks.landmark[30]
                    right = rslts.pose_landmarks.landmark[29]
                    left_hand = rslts.pose_landmarks.landmark[20]
                    right_hand = rslts.pose_landmarks.landmark[19]
                    left_eye = rslts.pose_landmarks.landmark[5]
                    right_eye = rslts.pose_landmarks.landmark[2]

                    head.x,head.y = head.x*width +xp,head.y*height+yp
                    body1.x,body1.y,body2.x,body2.y = body1.x*width+xp,body1.y*height+yp,body2.x*width+xp,body2.y*height+yp
                    left.x,left.y,right.x,right.y = left.x*width+xp,left.y*height+yp,right.x*width+xp,right.y*height+yp
                    left_hand.x,left_hand.y,right_hand.x,right_hand.y = left_hand.x*width+xp,left_hand.y*height+yp,right_hand.x*width+xp,right_hand.y*height+yp
                    left_eye.x,left_eye.y,right_eye.x,right_eye.y = left_eye.x*width+xp,left_eye.y*height+yp,right_eye.x*width+xp,right_eye.y*height+yp
                    

                    
                    # verifie si la personne courante porte un casque ou non
                if len(casques)>0 : 
                        for casque in casques:
                            csq = 0
                            print(casque)
                            xc, yc, wc, hc = casque[:4]
                            #cv2.rectangle(frame,(int(0),int(yc)),(int(wc),int(hc)),(0,255,0),2)
                            if ((hc > yp and xp > wc and hp > hc) and head.visibility > 0.1):
                               csq=1
                               break
                            else :
                               csq=0
                        if csq ==0: 
                               
                            cv2.putText(frame,'pas de casque',(int(xp),int(yp)+10),1,1.1,(0,255,0),1)
                            personnes_sans_casques.append(person)
                if len(casques)== 0 and head.visibility >0.1 :
                        personnes_sans_casques.append(person)
                        cv2.putText(frame,'pas de casque',(int(xp),int(yp)+10),1,1.1,(0,255,0),1)
                if len(blouses)>0 : 
                    
                    for blouse in blouses:
                        bls = 0
                        
                        xb, yb, wb, hb = blouse[:4]
                        
            
                        if ((xp < wb and hp > hb and yb <hp ) and (body1.visibility > 0.1 and body2.visibility > 0.1)):
                            bls =1
                            break
                        else : 
                            bls =0 
                    if bls ==0 :
                        personnes_sans_blouses.append(person)
                        cv2.putText(frame,'pas de blouse',(int(xp),int(yp)),1,1.1,(0,255,0),1)
                if len(blouses)==0 and body1.visibility>0.1 and body2.visibility>0.1:
                        cv2.putText(frame,'pas de blouse',(int(xp),int(yp)),1,1.1,(0,255,0),1)
                        personnes_sans_blouses.append(person)
                if len(soudages)>0 : 
                        for casque in soudages:
                            xc, yc, wc, hc = casque[:4]
                            #cv2.rectangle(frame,(int(xc),int(yc)),(int(wc),int(hc)),(0,255,0),2)

                            if ((hc > yp and xp > wc and hp > hc) and head.visibility > 0.1) :
                               personnes_sans_soudages.append(person)
                if len(soudages)==0 and head.visibility>0.3 : 
                        personnes_sans_soudages.append(person)
                if len(goggles)>0 : 
                        for goggle in goggles:
                            gog =0
                            x, y, w, h = goggle[:4]
                            #cv2.rectangle(frame,(int(x),int(y)),(int(w),int(h)),(0,255,0),2)

                            if (((left_eye.x >x and left_eye.x <w and left_eye.y <h and left_eye.y > y) or (right_eye.x >x and right_eye.x<w and right_eye.y < h and right_eye.y >y)) and right_eye.visibility>0.3 and left_eye.visibility>0.3) :
                                gog =1
                                break
                            else : 
                                gog =0
                        if gog ==0 : 
                            personnes_sans_goggles.append(person)
                            cv2.putText(frame,'pas de goggles',(int(xp),int(yp)+30),1,1.1,(0,255,0),1)
                if len(goggles)==0 and right_eye.visibility>0.3 and left_eye.visibility>0.1: 
                        personnes_sans_goggles.append(person)
                        cv2.putText(frame,'pas de goggles',(int(xp),int(yp)+30),1,1.1,(0,255,0),1)
                if len(gloves)>0 : 
                        for glove in gloves:
                            
                            glv = 0
                            x, y, w, h = glove[:4]
                            print(head.x,right_hand.x)

                            cv2.rectangle(frame,(int(x),int(y)),(int(w),int(h)),(0,255,0),2)
                            print(right_hand.visibility,left_hand.visibility)
                            if (((left_hand.x >x and left_hand.x<w and left_hand.y >y and left_hand.y<h) or (right_hand.x >x and right_hand.x<w and right_hand.y >y and right_hand.y<h)  ) and right_hand.visibility>0.4 and left_hand.visibility >0.3) :
                                glv =1
                                break
                            else :
                                glv =0
                        if glv ==0 : 
                                
                            personnes_sans_gloves.append(person)
                            cv2.putText(frame,'pas de gloves',(int(xp),int(yp)+20),1,1.1,(0,255,0),1)
                if len(gloves )==0 and left_hand.visibility>0.1 and right_hand.visibility>0.1 : 
                        print(f'glove  {right_hand.visibility}')
                        personnes_sans_gloves.append(person)
                        cv2.putText(frame,'pas de gloves',(int(xp),int(yp)+20),1,1.1,(0,255,0),1)

        # verifier si la personne porte des chaussures
        # verifier si ces deux points sont dans les boites englobantes des chaussures et s'ils sont visibles
                if len(chaussures)>0:
                        for chaussure in chaussures:
                            chauss =0
                            print('chaussure detectee')
                            x, y, w, h = chaussure[:4]
                            cv2.rectangle(frame,(int(x),int(y)),(int(w),int(h)),(0,0,250),2)

                            if (((left.x < w and left.x > x and left.y > y and left.y < h) or
                                (right.x < w and right.x > x and right.y > y and right.y < h)) and
                                right.visibility > 0.3 and left.visibility > 0.3): 
                                chauss =1
                                break 
                            else :
                                chauss =0
                        if chauss ==0 :                            
                            personnes_sans_chaussures.append(person)
                            cv2.putText(frame,'',(int(xp),int(yp)+40),1,1.1,(0,255,0),1)
                if len(chaussures)==0 and left.visibility>0.3 and right.visibility>0.3 : 
                        personnes_sans_chaussures.append(person)  
                        cv2.putText(frame,'pas de chaussures',(int(xp),int(yp)+40),1,1.1,(0,255,0),1)  
                    

                if all(person not in liste for liste in [personnes_sans_casques,personnes_sans_gloves,personnes_sans_chaussures,personnes_sans_blouses,personnes_sans_goggles]):
                        message = ''
                else:
                        
                        
                        cv2.rectangle(frame,(int(xp),int(yp)),(int(wp),int(hp)),(255,0,0),1)
                        messages=[]
                        if person  in personnes_sans_blouses : 
                            messages.append('La personne ne porte pas de blouse')
                            personnes_sans_EPI.append(person)
                            
                        if person  in personnes_sans_casques : 
                            messages.append('la personne ne porte pas de casque')
                            personnes_sans_EPI.append(person)
                            
                        if person  in personnes_sans_gloves : 
                            messages.append('la personne ne porte pas de gloves')
                            personnes_sans_EPI.append(person)
                            
                        if person  in personnes_sans_goggles : 
                            messages.append('la personne ne porte pas de goggles')
                            personnes_sans_EPI.append(person)
                            
                        if person  in personnes_sans_chaussures : 
                            messages.append('la personne ne porte pas de chaussures')
                            personnes_sans_EPI.append(person)
                            cv2.putText(frame,'pas de chaussures',(int(xp),int(yp)+40),1,1.1,(0,255,0),1)
                        
                              
                        
                        message = ', '.join(messages)
            
 

        
        
        # Convertissez l'image pour l'affichage dans Flask
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    frame_counter += 1
                    
                        
                    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    print("Failed to encode frame to JPEG")
            else:
                print("Invalid frame")
            
            



            if len(persons)==0 : 
                print('00000000000000000000000000000000000000000000')
        

@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/erreur_image')
def erreur_image():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')





@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == '__main__':
    

    # Démarrer le serveur Flask-SocketIO avec eventlet
    socketio.run(app, debug=True)


