#!/usr/bin/env python
# -*- coding: utf-8 -*-
# de Nuevo Control EIRL
import numpy as np
import pidfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import codecs
import obspy as obs
from obspy.clients.seedlink import Client as Clientseed
from obspy.clients.fdsn import Client as Clientfdsn
from obspy.signal.trigger import classic_sta_lta
from obspy import UTCDateTime
from obspy.io.xseed import Parser
from obspy.core import Stream
import datetime
from threading import Thread
import string
import os,sys
from dotenv import load_dotenv
from astropy.convolution import convolve, Gaussian1DKernel, Box1DKernel
import matplotlib.ticker as ticker
import subprocess
import pyrotd
from obspy.clients.fdsn.header import FDSNException
import urllib, json
import decimal
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
import pdfkit
import pandas as pd 
from scipy.interpolate import interp1d
import os.path
from db import connection_db
import time
import json
import io
import math

global Inventario,st,MyTiempo,stm,alat,alon,frecbajo,frecalto,esta,ancho,fileout1,fileout,utc,ProcessSpeed,ProcessDisplacement,groupth,fdsn_port
global sgatras,sgadelante,l1,l2,ProcessSpectrum,ProcessFourier,osc_damping,StreamFil,sismo,newestaciones,random,web_server,seedlink_port,server_ip,plantilla
global tiempoespera,fullpath,prefijo,Filtro,MyAccPlus

ProcessSpeed = True
ProcessDisplacement = True
ProcessFourier = True
ProcessSpectrum = True
newestaciones = False
plantilla   = False
Share   =   False

MyTiempo = 180
frecbajo = 0.10
frecalto = 25
ancho = 0.4
utc = -5 # En horas positivo o negativo
groupth = 10
atras = 10 # Tiempo en segundos para retroceder desde AHORA para extraer datos en tiempo real
sgatras = 10
sgadelante = 120
l1 = 1.5
l2 = 0.5
osc_damping = 0.05 #

#Funcion que recuepra el inventario de estaciones desde un xml
def GetInventory():
    global fullpath
    fResponse = fullpath+"/py/response.xml"
    inventario = obs.read_inventory(fResponse)
    return inventario

#Funcion que verifica las estaciones en linea
#def GetOnlineStations(inv):
#    global server_ip,seedlink_port
#    txtip = str(server_ip)+":"+str(seedlink_port)
    #print (txtip)
#    cmd = ['/home/sysop/seiscomp/bin/slinktool', '-Q', txtip]
#    mios = []
#    ahora = UTCDateTime()
#    lapso = 300 # 5 minutos de tolerancia
#    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
#    process.wait()
#    for line in process.stdout:
#        nys = line[0:8].strip()
#        hdr = UTCDateTime(line.split("-")[1])
#        dem = ahora - hdr
#        if dem < lapso:
#            b = nys in mios
#            if b == False:
#                mios.append(nys)
#    return mios
def GetOnlineStations(inv):
    global server_ip,seedlink_port,path_seiscomp
    #cmd = ['/home/seis/seiscomp3/bin/slinktool', '-Q', 'localhost:18000']
    cmd = [path_seiscomp+'bin/slinktool', '-Q', str(server_ip)+':'+str(seedlink_port)]
    mios = []
    ahora = UTCDateTime()
    tolerancia = 300 # 5 minutos de tolerancia
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    process.wait()
    logs("* * * * * Leyendo estaciones disponibles... * * * * *")
    for line in process.stdout:
        line = line.decode('utf-8')
        nys = line[0:2]+"."+line[3:8].replace(" ","")+"."+line[9:11].strip()+"."
        hdr = UTCDateTime(line.split(" - ")[-1])
        dem = ahora - hdr
        if dem < tolerancia:
            b = nys in mios
            if b == False:
                mios.append(nys)
    return mios

#Funcion que devuelve la signal via seedlink
def GetStreamSeedLink(Evento):
    global stm
    #n = len(Inventario.get_contents()['stations'])
    MyStream = obs.Stream()

    t1 = UTCDateTime(Evento['FechayHora'])
    t2 = t1 + Evento['Tiempo']

    stationsOnline = GetOnlineStations(Inventario)
    n = len(stationsOnline)
    hilos = [0]*n
    stm   = [0]*n

    mini = 0 
    logs ("Hay "+str(n)+" estaciones para adquirir registro desde las: "+str(t1)+" hasta las: "+str(t2)+"\n")
    logs ("Pero solo hay estas estaciones en linea:")
    logs (str(stationsOnline))

    if(Evento['NewEstaciones'] != False):
        logs ("Solo se va procesar los pedido por el usuario:")
        stationsOnline = CheckStationEnLinea(stationsOnline,Evento)
        logs (str(stationsOnline))
        n = len(stationsOnline)

    if n > groupth:
        nm = n//groupth + 1
        #nr = n%groupth
        nn = groupth
    else:
        nm = 1
        nn = n

    #print("n: "+str(n)+" / nm: "+str(nm))
    for y in range(0,nm):
        logs ("Ejecutando bloque ("+str(y+1)+"): Contiene "+str(nn - mini)+" estaciones...\n")
        yy = y + 1
        if yy == nm:
            ff = n
        else:
            ff = yy*nn

        xv1 = []
        for x in range(mini, ff):
            network,station,location,channel = stationsOnline[x].split(".")
            channel = Inventario.select(network=network,station=station,location=location).get_contents()['channels'][0].split(".")[3][:-1]+'?'
            xv1.append(x)
            #logs(str(network)+"/"+str(station)+"/"+str(location)+"/"+str(channel)+"/")
            hilos[x] = Thread(target=ThreadGetStream, args=(stm,x,network,station,location,channel,t1,t2))
            hilos[x].start()

        xv2 = []
        for x in xv1: 
            if(hilos[x]):
                xv2.append(x)
                hilos[x].join()

        for x in xv2:
            
            try:
                isi = len(stm[x])
                #consulto si tiene 3 canales
                #print("stm:")
                #print(stm[x])
                if isi == 3:
                    MyStream += stm[x]
            except AssertionError as error:
                logs(error)

        mini = ff

    tmpn = len(MyStream)
    logs ("1 Se tienen un total de "+str(tmpn)+" registros por procesar.\n")
    return MyStream

#
def CheckStationEnLinea(stationsOnline,Evento):
    nuevas = Evento['NewEstaciones'].split(":")
    #ni = len(nuevas)
    #nj = len(stationsOnline)
    caset = []
    #ww = [i.split(' ', 1)[1] for i in stationsOnline]
    #print(str(ww))
    for xtmp in nuevas:
        #xtmp = nuevas[xi]
        vu = xtmp in stationsOnline
        if (vu == True):
            caset.append(xtmp)
    return caset

#Funcion que devuelve la signal via arclink
def GetStreamArcLink(Evento):
    global stm,server_ip,fdsn_port
    #n = len(Inventario.get_contents()['stations'])
    MyStream = obs.Stream()
    client = Clientfdsn('http://'+server_ip+':'+fdsn_port)

#    print(str(client))
    if Evento['FechayHora'] is False:
        t2 = UTCDateTime() - atras
        t1 = t2 - MyTiempo
    else:
        t1 = UTCDateTime(Evento['FechayHora'])
        t2 = t1 + Evento['Tiempo']

    #print str(t1)+"/"+str(t2)
    if(Evento['NewEstaciones'] != False):
        misestaciones = Evento['NewEstaciones'].split(":")
        nes = len(misestaciones)
        for xi in range(0, nes):
            #print("estaciones: "+str(xi))
            tmpred,tmpesta,tmploc,tmpcha = misestaciones[xi].split(".")

            stm_tmp = obs.Stream()
            try:
                stm_tmp  += client.get_waveforms(tmpred, tmpesta, tmploc, "HH*", t1, t2)  # Canales para Reftek
                MyStream += stm_tmp.select(channel="HHN")
                MyStream += stm_tmp.select(channel="HHE")
                MyStream += stm_tmp.select(channel="HHZ")
                mrHH = True
            except FDSNException:
                mrHH = False

            stm_tmp = obs.Stream()
            try:
                stm_tmp += client.get_waveforms(tmpred, tmpesta, tmploc, "HN*", t1, t2) # Canales para kinemetrics
                MyStream += stm_tmp.select(channel="HNN")
                MyStream += stm_tmp.select(channel="HNE")
                MyStream += stm_tmp.select(channel="HNZ")
                mrHN = True
            except FDSNException:
                mrHN = False

            stm_tmp = obs.Stream()
            try:
                stm_tmp += client.get_waveforms(tmpred, tmpesta, tmploc, "EN*", t1, t2) # Canales para Raspberry Shake
                MyStream += stm_tmp.select(channel="ENN")
                MyStream += stm_tmp.select(channel="ENE")
                MyStream += stm_tmp.select(channel="ENZ")
                mrEN = True
            except FDSNException:
                mrEN = False

            stm_tmp = obs.Stream()
            try:
                stm_tmp += client.get_waveforms(tmpred, tmpesta, tmploc, "BH*", t1, t2) # Canales para Guralp
                MyStream += stm_tmp.select(channel="BHN")
                MyStream += stm_tmp.select(channel="BHE")
                MyStream += stm_tmp.select(channel="BHZ")
                mrEN = True
            except FDSNException:
                mrEN = False

    else:
        stm_tmp = obs.Stream()
        try:
            stm_tmp += client.get_waveforms("??", "*", "**", "HH*", t1, t2)  # Canales para Reftek
            MyStream += stm_tmp.select(channel="HHN")
            MyStream += stm_tmp.select(channel="HHE")
            MyStream += stm_tmp.select(channel="HHZ")
            mrHH = True
        except FDSNException:
            mrHH = False

        stm_tmp = obs.Stream()
        try:
            stm_tmp += client.get_waveforms("??", "*", "**", "HN*", t1, t2) # Canales para kinemetrics
            MyStream += stm_tmp.select(channel="HNN")
            MyStream += stm_tmp.select(channel="HNE")
            MyStream += stm_tmp.select(channel="HNZ")
            mrHN = True
        except FDSNException:
            mrHN = False

        stm_tmp = obs.Stream()
        try:
            stm_tmp += client.get_waveforms("??", "*", "**", "EN*", t1, t2) # Canales para Raspberry Shake
            MyStream += stm_tmp.select(channel="ENN")
            MyStream += stm_tmp.select(channel="ENE")
            MyStream += stm_tmp.select(channel="ENZ")
            mrEN = True
        except FDSNException:
            mrEN = False
        
        stm_tmp = obs.Stream()
        try:
            stm_tmp += client.get_waveforms("??", "*", "**", "BH*", t1, t2) # Canales para Guralp
            MyStream += stm_tmp.select(channel="BHN")
            MyStream += stm_tmp.select(channel="BHE")
            MyStream += stm_tmp.select(channel="BHZ")
            mrEN = True
        except FDSNException:
            mrEN = False

    tmpn = len(MyStream)
    logs ("2 Se tienen un total de "+str(tmpn)+" registros por procesar.\n")
    return MyStream

#Funccion para descargar el stream de una estacion
def ThreadGetStream(result,index,network,station,location,channel,t1,t2):
    global server_ip,seedlink_port,tiempoespera,obs
    client = Clientseed(str(server_ip),int(seedlink_port),timeout=5)
    txtvar = str(network)+"/"+str(station)+"/"+str(location)+"/"+str(channel)+"/"+str(t1)+"/"+str(t2)
    logs(txtvar)
    tmpstr = client.get_waveforms(network, station, location, channel, t1, t2)
    newtmpst = obs.Stream()
    newtmpst += tmpstr.select(channel="ENN")
    newtmpst += tmpstr.select(channel="ENE")
    newtmpst += tmpstr.select(channel="ENZ")
    newtmpst += tmpstr.select(channel="HHN")
    newtmpst += tmpstr.select(channel="HHE")
    newtmpst += tmpstr.select(channel="HHZ")
    newtmpst += tmpstr.select(channel="HNN")
    newtmpst += tmpstr.select(channel="HNE")
    newtmpst += tmpstr.select(channel="HNZ")
    newtmpst += tmpstr.select(channel="BHN")
    newtmpst += tmpstr.select(channel="BHE")
    newtmpst += tmpstr.select(channel="BHZ")

    #for tr in tmpstr.select(component="C"):
    #    tmpstr.remove(tr)
    #for tr in tmpstr.select(component="S"):
    #    tmpstr.remove(tr)
    #for tr in tmpstr.select(component="L"):
    #    tmpstr.remove(tr)
    newtmpst.merge(fill_value='latest')
    logs(str(newtmpst))
    result[index] = newtmpst
    #print(str(result[index]))

#Funcion para validad fecha y hora
def ValidateDT(date_text):
    if date_text is False:
        return False
    try:
        datetime.datetime.strptime(date_text, '%Y-%m-%dT%H:%M:%SZ')
        mr = True
    except ValueError:
        mr = False
    return mr

#Funcion para uniformizar la signal
def uniformizo_streams(n,Evento,stn1):
    global st
    stg = obs.Stream()
    s = 0
    zmini = 0
    zmaxi = 0
    zminf = 0
    zmaxf = 0

    for x in range(0, n):
        i0 = s
        i1 = s + 1
        i2 = s + 2
        s  = i2 + 1
        
        tt0 = stn1[i0].stats.starttime
        tt1 = stn1[i1].stats.starttime
        tt2 = stn1[i2].stats.starttime
        fmini = min(tt0,tt1,tt2)
        fmaxi = max(tt0,tt1,tt2)

        t1 = UTCDateTime(Evento['FechayHora'])
        ultimonombre = ""

        if chech_error(fmini,fmaxi,t1):
            continue

        ff0 = stn1[i0].stats.endtime
        ff1 = stn1[i1].stats.endtime
        ff2 = stn1[i2].stats.endtime
        fminf = min(ff0,ff1,ff2)
        fmaxf = max(ff0,ff1,ff2)

        t2 = t1 + Evento['Tiempo']
        if chech_error(fminf,fmaxf,t2):
            continue

        if x == 0:
            zmini = fmini
            zmaxi = fmaxi
            zminf = fminf
            zmaxf = fmaxf
        else:
            if fmini < zmini:
                zmini = fmini
            if fmaxi > zmaxi:
                zmaxi = fmaxi
            if fminf < zminf:
                zminf = fminf
            if fmaxf > zmaxf:
                zmaxf = fmaxf
    
        stg += stn1[i0]
        stg += stn1[i1]
        stg += stn1[i2]

    zmini = zmini - 1
    zmaxi = zmaxi + 1
    stg.cutout(zmini,zmaxi)
    zminf = zminf - 1
    zmaxf = zmaxf + 1
    stg.cutout(zminf,zmaxf)

    return stg

#Funcion que solo deja 3 canales de toda la trama
def solo3channels(myst):
    c = 0
    name = myst[0].stats.station
    location = myst[0].stats.location
    tmpstream = obs.Stream()
    totstream = obs.Stream()
    u = 0
    for st1 in myst:
        myname = st1.stats.station
        mylocation = st1.stats.location
        mychannel = st1.stats.channel
        eschannel = verificaletra(mychannel)
        if (name == myname and location == mylocation):
            if (eschannel == True):
                c += 1
                tmpstream += st1
                #print(" Nombre: "+str(myname)+" Canal: "+str(mychannel))
        else:
            tmpstream = obs.Stream()
            tmpstream += st1
            c = 1
            #print(" Nombre: "+str(myname)+" Canal: "+str(mychannel))

        name = myname
        location = mylocation
        
        if (c == 3):
            totstream += tmpstream

    return totstream

def verificaletra(canal):
    resp = False
    letra = canal[-1]
    if letra == 'Z':
        resp = True
    if letra == 'N':
        resp = True
    if letra == 'E':
        resp = True
    return resp

#Funcion que busca un error en la trama de datos
def chech_error(h1,h2,horayfecha):
    lapso = 5 # margen de error en segundos al inio o final de la trama
    vtmp1 = abs(horayfecha - h1)
    if vtmp1 > lapso:
        return True
    vtmp2 = abs(horayfecha - h2)
    if vtmp2 > lapso:
        return True

    return False

#Funcion para leer argumentos de linea de comandos
def GetArg(MyArg):
    global newestaciones,MyTiempo,plantilla
    detectP = False
    narg = len(MyArg)
    NewCode = False
    NewEmail = False
    NewPDF = False
    Tiempo     = float(MyTiempo)
    Fuente  = False
    Respuesta = False
    Filtro = False
    Plantilla = False
    Share = False

    for x in MyArg:
        logs (str(x))
        xtmp = x.split("=")
        xcode = xtmp[0]
        nxc = len(xtmp)
        if nxc == 2:
            xdata = xtmp[1]
        else:
            xdata = False

        if xcode == "M":
            Modo = xdata

        if xcode == "F":
            FechayHora = xdata

        if xcode == "T":
            Tiempo = float(xdata)

        if xcode == "I":
            Id = int(xdata)

        if xcode == "P":
            detectP = True

        if xcode == "E":
            newestaciones = xdata

        if xcode == "N":
            NewCode = True

        if xcode == "R":
            NewEmail = True
        
        if xcode == "O":
            NewPDF = True
        
        if xcode == "X":
            Respuesta = True
        
        if xcode == "Z":
            Filtro = True

        if xcode == "S":
            Fuente = float(xdata)
        
        if xcode == "L":
            Plantilla = xdata
        
        if xcode == "G":
            Share = True

    if ValidateDT(FechayHora) is False:
        logs("Mal Hora")
        exit()

    if (UTCDateTime() - UTCDateTime(FechayHora)) < 0:
        logs ("Fecha y Hora no puede ser futuro.")
        exit()

    Evento = {
        'Modo'          :   Modo,
        'FechayHora'    :   FechayHora,
        'Tiempo'        :   Tiempo,
        'Id'            :   Id,
        'DetectP'       :   detectP,
        'NewEstaciones' :   newestaciones,
        'NewCode'       :   NewCode,
        'NewEmail'      :   NewEmail,
        'NewPDF'        :   NewPDF,
        'Fuente'        :   Fuente,
        'Respuesta'     :   Respuesta,
        'Filtro'        :   Filtro,
        'Plantilla'     :   Plantilla,
        'Share'         :   Share
    }
    return Evento
    
#Funcion que trae el nombre real de una estacion desde el inventario
def get_nombre_real(estacion,ninv):
    ms = ninv.get_contents()['stations']
    ret_name = [s for s in ms if estacion in s][0].split(" ",1)[1][1:-1]
    return ret_name

#Funcion para crear cadera random de 10 caracteres
def random_key(length):
    import random
    import string
    #key = ''
    #for i in range(length):
    #    key += random.choice(string.lowercase + string.uppercase + string.digits)
    #return key
    letters = string.ascii_lowercase + string.ascii_uppercase + string.digits
    return ''.join(random.choice(letters) for i in range(length))

#Funcion que saca parametros de sismo desde db mysql
def GetParametros(Evento):
    #import psycopg2
    global random,sismo,prefijo
    #import mysql.connector
    myId = Evento["Id"]
    if(myId == False):
        sismo = {'gid': False,'fecha': '* * Sin vinculacion * *','hora': '* * Sin vinculacion * *','magnitud': '* * Sin vinculacion * *','intensidad': '* * Sin vinculacion * *','longitud': '* * Sin vinculacion * *','latitud': '* * Sin vinculacion * *','profundidad': '* * Sin vinculacion * *','fuente':'* * Sin vinculacion * *','sourceigp':'0'}
    else:
        logs ("Conectando al servidor local")

        #connection = mysql.connector.connect(host='localhost', database='joomla', user='sysop', password='sysop',auth_plugin='mysql_native_password')
        connection = connection_db()
        if connection:
            #cur = conn.cursor()
            cursor = connection.cursor()
            if Evento["NewCode"] == True:
                sql1 = "UPDATE "+prefijo+"ncnsismos_sismosigp SET code = '"+random+"' WHERE id = '"+str(myId)+"';"
                #logs("UPDATE: "+sql1)
                #print (str(sql1))
                cursor.execute(sql1)
                #connection.commit()
            
            
            #cur.execute(sql1)
            #sql1 = "SELECT gid,fechalocal,horalocal,magnitud,intensidad,st_x(the_geom) AS lon,st_y(the_geom) AS lat,profundidad FROM sismosigp WHERE gid = '"+str(myId)+"';"
            sql2 = "SELECT id,fecha,magnitud,intensidad,longitud,latitud,profundidad,code,sourceigp FROM "+prefijo+"ncnsismos_sismosigp WHERE id = '"+str(myId)+"';"
            #logs("SELECT: "+sql2)
            #cur.execute(sql1)
            #print (str(sql2))
            #cursor = connection.cursor()
            cursor.execute(sql2)
            #connection.commit()
            result = cursor.fetchall()
            #logs(str(result))
            

            #operation = sql1 + ";" + sql2
            #cursor = connection.cursor()
            #for result in cursor.execute(operation, multi=True):
            #    result = cursor.fetchall()
                #print str(result)
            
            param = result[0]
            tmpfecha = str(param[1]).split(" ")
            sismo = {'id':str(param[0]),'fecha':str(tmpfecha[0]),'hora':str(tmpfecha[1]),'magnitud':str(param[2]),'intensidad':str(param[3].encode('latin1')),'longitud':str(param[4]),'latitud':str(param[5]),'profundidad':str(param[6]),'fuente':'IGP','code':str(param[7]),'sourceigp':str(param[8])}
            connection.commit()
            cursor.close()
            connection.close()
        else:
            logs ("No hay conexion al servidor. Verifique.")
            sismo = False
    return sismo

#Detect wave P
def detectWaveP(wave):
    mydf = wave.stats.sampling_rate
    mycft = classic_sta_lta(wave.data, int(5 * mydf), int(10 * mydf))
    hinicio = wave.stats.starttime
    hfinal  = wave.stats.endtime
    
    try:
        myres1 = next(x for x, val in enumerate(mycft) if val > l1)
        mh1 = wave.stats.starttime + myres1/mydf - sgatras
    except StopIteration:
        myres1 = 0
        mh1 = hinicio - 1

    #myh1  = hinicio + myres1/200.0
    try:
        myres2 = myres1 + next(x for x, val in enumerate(mycft[myres1:]) if val < l2)
        mh2 = wave.stats.starttime + myres2/mydf + sgadelante
    except StopIteration:
        mh2 = hfinal + 1

    if mh1 < hinicio:
        mh1 = hinicio
        print ("Inicio muy atras, dejando igual")
    
    if mh2 > hfinal:
        mh2 = hfinal
        print ("Final muy adelante, dejando igual")
        
    #stg.cutout(zmini,zmaxi)
    return [mh1,mh2]

#Cut wave
def cortoWaveP(wave,a,b):
    totalinicio = wave[0].stats.starttime
    totalfinal  = wave[0].stats.endtime

    if totalinicio < a:
        wave.cutout(totalinicio - 1,a + 1)
        print ("Corto Ok adelante")
        
    if totalfinal > b:
        wave.cutout(b - 1,totalfinal + 1)
        print ("Corto OK al final")

    return wave

#Funcion que procesa un evento con un stream
def FileProcess(Evento,Inventario):
    logs("-- inicio FileProcess --")
    import shutil
    import pwd
    global alat,alon,frecbajo,frecalto,esta,ancho,fileout1,utc,StreamFil,sismo,st,random,fileout,MyAccPlus
    StreamFil = obs.Stream()
    n = int(st.count()/3)
    s = 0
    uestacion = ""
    
    alat = []
    alon = []
    pgae = []
    pgan = []
    pgaz = []
    esta = []
    accZ = {}
    accN = {}
    accE = {}

    contador = 0
    
    logs("Llevamos el random: "+str(random))
    tmp_random = random

    sismo = GetParametros(Evento) # Tambien actualiza con el NewCode del evento

    if Evento["NewCode"] == True:
        random = tmp_random
    else:
        random = sismo['code']

    MyAcc = {random:{}}
    MyAccPlus = {random:{}}
    logs("Nos quedamos con el random: "+str(random))
    myuser = pwd.getpwuid( os.getuid() ).pw_name

    fileout1 = fileout+str(random)+"/"
    logs("Escribiendo en: "+str(fileout1)+ " y se establecen permisos de escritura para: "+str(myuser))
    try:
        logs("inicio crear carpeta en el events: "+str(random))
        os.mkdir( fileout1[:-1] , 0o777)
        logs("fin crear carpeta en el events: "+str(random))
    except Exception as e:
        logs("Error: "+str(e))
        logs("Ya existe el directorio \""+random+"\", estamos en actualizacion de eventos. Se procede a borrar su contenido previo.")
        shutil.rmtree(fileout1[:-1], ignore_errors=True)
        os.mkdir( fileout1[:-1] , 0o777)
        #for f in os.listdir(fileout1):
            #os.remove(os.path.join(fileout1, f))
    
    #os.chmod( fileout1 , 0o0777)
    
    logs("Existen "+str(n)+" estaciones con datos\n")
    fakenet = []
    for x in range(0, n):
        i0 = s          #0,3,6
        i1 = s + 1      #1,4,7
        i2 = s + 2      #2,5,8... 
        s  = i2 + 1

        #Frecuencia de muestreo
        fps = st[i0].stats.sampling_rate
        #Juntar coordenadas de estaciones
        txtation = st[i0].stats.network+"."+st[i0].stats.station+"."+st[i0].stats.location+"."+st[i0].stats.channel
        logs ("\nTrabajando con "+txtation)
        punto = Inventario.get_coordinates(txtation)
        xlat = punto['latitude']
        xlon = punto['longitude']
        alat.append(xlat)
        alon.append(xlon)
             
        st[i0].detrend("linear")
        st[i0].detrend("demean")
        st[i1].detrend("linear")
        st[i1].detrend("demean")
        st[i2].detrend("linear")
        st[i2].detrend("demean")

        tr0 = st[i0]
        tr1 = st[i1]
        tr2 = st[i2]
        ## Filtering with a lowpass on a copy of the original Trace
        filt0 = tr0.copy()
        filt1 = tr1.copy()
        filt2 = tr2.copy()

        #Filtros
        #filt0.filter('highpass', freq=0.5, corners=2, zerophase=True)
        if Evento["Filtro"] == True:
            filt0 = filt0.filter('bandpass', freqmin=frecbajo,freqmax=frecalto)
            filt1 = filt1.filter('bandpass', freqmin=frecbajo,freqmax=frecalto)
            filt2 = filt2.filter('bandpass', freqmin=frecbajo,freqmax=frecalto)
            msg_fil = "Filtro pasa banda ("+str(frecbajo)+"~"+str(frecalto)+" Hz)"
            xfil = 0.73
        else:
            msg_fil = "Sin filtro"
            xfil = 0.93

        if Evento['DetectP'] == True:
            #Aqui tengo q buscar la onda P y el final para cortar la señal.
            logs ("Detectando ondas P")
            [ nuevoinicio , nuevofin ] = detectWaveP(filt2)
        else:
            logs ("Paso de frente con toda la signal")
            nuevoinicio = filt2.stats.starttime
            nuevofin    = filt2.stats.endtime

        logs ("Inicio normal: "+str(st[i0].stats.starttime))
        logs ("Inicio detect: "+str(nuevoinicio))
        logs ("Final normal : "+str(st[i0].stats.endtime))
        logs ("Final detect : "+str(nuevofin))

        #construyo Stream de 3 canales con filtro para cortarlos (Cambio filt0,1,2)
        tmpStream1 = Stream(traces=[filt0, filt1, filt2])
        logs ("Corto signal filtrada")
        newStream1 = cortoWaveP(tmpStream1,nuevoinicio,nuevofin)
        filt0 = newStream1[0]
        filt1 = newStream1[1]
        filt2 = newStream1[2]
        StreamFil += filt0
        StreamFil += filt1
        StreamFil += filt2

        #construyo Stream de 3 canales sin filtro para cortarlos (Cambio tr0,1,2)
        tmpStream2 = Stream(traces=[tr0, tr1, tr2])
        logs ("Corto signal en bruto")
        newStream2 = cortoWaveP(tmpStream2,nuevoinicio,nuevofin)
        tr0 = newStream2[0]
        tr1 = newStream2[1]
        tr2 = newStream2[2]

        #tittemp = UTCDateTime(st[i0].stats.starttime))
        tittemp = str(UTCDateTime(filt0.stats.starttime) + utc*3600).split("T")
        titulo_hora  = "Fecha: " + tittemp[0] + " / Hora: " + tittemp[1][0:8] + " UTC: " + str(utc) 

        t0 = (1.0/fps)*np.arange(0, len(tr0), 1)
        t1 = (1.0/fps)*np.arange(0, len(tr1), 1)
        t2 = (1.0/fps)*np.arange(0, len(tr2), 1)
            
        estacion    = st[i0].stats.station
        network     = st[i0].stats.network
        location    = st[i0].stats.location
        channel     = st[i0].stats.channel

        if (network in fakenet) is False:
            MyAcc[random].update({network:{}})
            MyAccPlus[random].update({network:{}})
            fakenet.append(network)

        #asigno nombre de estacion a diccionario solo una vez cada 3 canales
        p3 = i0%3
        #if p3 == 0:
        if uestacion != estacion:
            MyAcc[random][network].update({estacion:{}})
            MyAccPlus[random][network].update({estacion:{}})
        MyAcc[random][network][estacion].update({location:{}})
        MyAccPlus[random][network][estacion].update({location:{}})

        #esta.append(estacion)

        tmp_station = network+"."+estacion
        nombre_estacion = get_nombre_real(tmp_station,Inventario)
        
        pga  = [0,0,0]
        pgv  = [0,0,0]
        pgd  = [0,0,0]
        pgaf = [0,0,0]
        
        ##### Primer canal Aceleracion filtrado
        fig = plt.figure(figsize=(10, 8))
        ax0 = fig.add_subplot(311)
        ax0.set_title('[ACELERACION] '+network+'.'+estacion+'.'+location+' / '+str(titulo_hora))
        sy0 = filt0.data
        sy0 = 100*sy0
        accE[contador] = sy0
        
        ##### Primer canal Aceleracion sin filtrar
        zy0 = tr0.data
        zy0 = 100*zy0
        sinf0 = tr0.data*100
        varo = orientacion(st[i0].get_id())
        ax0.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax0.transAxes,color='k', fontsize=10)
        #max0 = round(sy0.max(),4)
        max0 = sy0.max()
        #max0n = round(sy0.min(),4)
        max0n = sy0.min()
        if abs(max0n) > max0:
            max0 = max0n
        
        pga[0] = max0
        if abs(max0) < 0.01:
            max0t = format(max0, '.4f')
        else:
            max0t = format(max0, '.2f')

        #max0f =round(zy0.max(),4)
        max0f = zy0.max()
        #max0fn = round(zy0.min(),4)
        max0fn = zy0.min()
        if abs(max0fn) > max0f:
            max0f = max0fn
        pgaf[0] = max0f
        pgae.append(abs(max0f))
        ax0.text(0.81, 0.95,'PGA: '+str(max0t)+' cm/s\u00B2',horizontalalignment='left',verticalalignment='top',transform = ax0.transAxes)
        plt.plot(t0, sy0,'b',linewidth=ancho)
        plt.grid()
        
        ##### Segundo canal Aceleracion filtrado
        ax1 = fig.add_subplot(312, sharex=ax0)
        sy1 = filt1.data
        sy1 = 100*sy1
        accN[contador] = sy1
        
        ##### Segundo canal Aceleracion sin filtrar
        zy1 = tr1.data
        zy1 = 100*zy1
        sinf1 = tr1.data*100
        varo = orientacion(st[i1].get_id())
        ax1.text(0.01, 0.95, varo,verticalalignment='top', horizontalalignment='left',transform=ax1.transAxes,color='k', fontsize=10)
        #max1 = round(sy1.max(),4)
        max1 = sy1.max()
        #max1n = round(sy1.min(),4)
        max1n = sy1.min()
        if abs(max1n) > max1:
            max1 = max1n

        pga[1] = max1
        if abs(max1) < 0.01:
            max1t = format(max1, '.4f')
        else:
            max1t = format(max1, '.2f')
        
        #max1f =round(zy1.max(),4)
        max1f = zy1.max()
        #max1fn = round(zy1.min(),4)
        max1fn = zy1.min()
        if abs(max1fn) > max1f:
            max1f = max1fn
        pgaf[1] = max1f
        pgan.append(abs(max1f))
        ax1.text(0.81, 0.95,'PGA: '+str(max1t)+' cm/s\u00B2',horizontalalignment='left',verticalalignment='top',transform = ax1.transAxes)
        plt.plot(t1, sy1,'g',linewidth=ancho)
        plt.ylabel('Aceleracion [cm/s\u00B2]')
        plt.grid()
        
        ##### Tercer canal Aceleracion filtrado
        ax2 = fig.add_subplot(313, sharex=ax0)
        sy2 = filt2.data
        sy2 = 100*sy2
        accZ[contador] = sy2
        
        ##### Tercer canal Aceleracion sin filtrar
        zy2 = tr2.data
        zy2 = 100*zy2
        sinf2 = tr2.data*100
        varo = orientacion(st[i2].get_id())
        ax2.text(0.01, 0.95, varo,verticalalignment='top', horizontalalignment='left',transform=ax2.transAxes,color='k', fontsize=10)
        #max2 = round(sy2.max(),4)
        max2 = sy2.max()
        #max2n = round(sy2.min(),4)
        max2n = sy2.min()
        if abs(max2n) > max2:
            max2 = max2n

        pga[2] = max2
        if abs(max2) < 0.01:
            max2t = format(max2, '.4f')
        else:
            max2t = format(max2, '.2f')
        
        #max2f =round(zy2.max(),4)
        max2f = zy2.max()
        #max2fn = round(zy2.min(),4)
        max2fn = zy2.min()
        if abs(max2fn) > max2f:
            max2f = max2fn
        pgaf[2] = max2f              
        pgaz.append(abs(max2f))
        ax2.text(0.81, 0.95,'PGA: '+str(max2t)+' cm/s\u00B2',horizontalalignment='left',verticalalignment='top',transform = ax2.transAxes)
        plt.plot(t2, sy2,'r',linewidth=ancho)
        plt.xlabel('Tiempo [s]')
        ax2.text(xfil,-0.16,msg_fil,verticalalignment='top', horizontalalignment='left',color='k', fontsize=10,transform = ax2.transAxes)
        plt.grid()
        
        contador = contador + 1
        
        #Genera archivos Aceleracion
        tmpfile     = fileout1+"RED_"+network+"_"+estacion+"_"+location+"_ACC.png"
        fig.savefig(tmpfile, format='png',bbox_inches='tight')
        os.chmod( tmpfile , 0o0777)
        #listafile.append(tmpfile)

        ############################################## Para unidad de g: /981 ##############################################
        ##### Primer canal Aceleracion filtrado (dividido por 9.81)
        fig2 = plt.figure(figsize=(10, 8))
        ax0_2 = fig2.add_subplot(311)
        ax0_2.set_title('[ACELERACION] '+network+'.'+estacion+'.'+location+' / '+str(titulo_hora))
        varo = orientacion(st[i0].get_id())
        ax0_2.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax0_2.transAxes,color='k', fontsize=10)
        max0g = float(max0) / 981.0
        
        if abs(max0g) < 0.01:
            max0gt = format(max0g, '.4f')
        else:
            max0gt = format(max0g, '.2f')

        ax0_2.text(0.81, 0.95,'PGA: '+str(max0gt)+' g',horizontalalignment='left',verticalalignment='top',transform = ax0_2.transAxes)
        sy0_2 = filt0.data * 100.0 / 981.0  # Multiplicar por 100 prara tener gal
        plt.plot(t0, sy0_2, 'b', linewidth=ancho)
        plt.grid()

        ##### Segundo canal Aceleracion filtrado (dividido por 9.81)
        ax1_2 = fig2.add_subplot(312, sharex=ax0_2)
        varo = orientacion(st[i1].get_id())
        ax1_2.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax1_2.transAxes,color='k', fontsize=10)
        max1g = float(max1) / 981.0
        
        if abs(max1g) < 0.01:
            max1gt = format(max1g, '.4f')
        else:
            max1gt = format(max1g, '.2f')

        ax1_2.text(0.81, 0.95,'PGA: '+str(max1gt)+' g',horizontalalignment='left',verticalalignment='top',transform = ax1_2.transAxes)
        sy1_2 = filt1.data * 100.0 / 981.0  # Multiplicar por 100 tener gal
        plt.plot(t1, sy1_2, 'g', linewidth=ancho)
        plt.ylabel('Aceleracion [g]')
        plt.grid()

        ##### Tercer canal Aceleracion filtrado (dividido por 9.81)
        ax2_2 = fig2.add_subplot(313, sharex=ax0_2)
        varo = orientacion(st[i2].get_id())
        ax2_2.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax2_2.transAxes,color='k', fontsize=10)
        max2g = float(max2) / 981.0
        
        if abs(max2g) < 0.01:
            max2gt = format(max2g, '.4f')
        else:
            max2gt = format(max2g, '.2f')
        
        ax2_2.text(0.81, 0.95,'PGA: '+str(max2gt)+' g',horizontalalignment='left',verticalalignment='top',transform = ax2_2.transAxes)
        sy2_2 = filt2.data * 100.0 / 981.0  # Multiplicar por 100 para tener gal
        plt.plot(t2, sy2_2, 'r', linewidth=ancho)
        plt.xlabel('Tiempo [s]')
        ax2_2.text(xfil, -0.16, msg_fil, verticalalignment='top', horizontalalignment='left', color='k', fontsize=10, transform=ax2_2.transAxes)
        plt.grid()

        # Generar archivo Aceleracion (multiplicado por 100)
        tmpfile2 = fileout1 + "RED_" + network + "_" + estacion + "_" + location + "_ACCg.png"
        fig2.savefig(tmpfile2, format='png', bbox_inches='tight')
        os.chmod(tmpfile2, 0o0777)
        ######################################################################################################################

        ############################################## Para unidad de m: /100 ##############################################
        ##### Primer canal Aceleracion filtrado (dividido por 100)
        fig2 = plt.figure(figsize=(10, 8))
        ax0_2 = fig2.add_subplot(311)
        ax0_2.set_title('[ACELERACION] '+network+'.'+estacion+'.'+location+' / '+str(titulo_hora))
        varo = orientacion(st[i0].get_id())
        ax0_2.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax0_2.transAxes,color='k', fontsize=10)
        max0g = float(max0) / 100.0
        
        if abs(max0g) < 0.01:
            max0gt = format(max0g, '.4f')
        else:
            max0gt = format(max0g, '.2f')

        ax0_2.text(0.81, 0.95,'PGA: '+str(max0gt)+' m/s\u00B2',horizontalalignment='left',verticalalignment='top',transform = ax0_2.transAxes)
        sy0_2 = filt0.data  # Ya esta en metros
        plt.plot(t0, sy0_2, 'b', linewidth=ancho)
        plt.grid()

        ##### Segundo canal Aceleracion filtrado (dividido por 9.81)
        ax1_2 = fig2.add_subplot(312, sharex=ax0_2)
        varo = orientacion(st[i1].get_id())
        ax1_2.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax1_2.transAxes,color='k', fontsize=10)
        max1g = float(max1) / 100.0
        
        if abs(max1g) < 0.01:
            max1gt = format(max1g, '.4f')
        else:
            max1gt = format(max1g, '.2f')

        ax1_2.text(0.81, 0.95,'PGA: '+str(max1gt)+' m/s\u00B2',horizontalalignment='left',verticalalignment='top',transform = ax1_2.transAxes)
        sy1_2 = filt1.data  # Ya está en metros
        plt.plot(t1, sy1_2, 'g', linewidth=ancho)
        plt.ylabel('Aceleracion [m/s\u00B2]')
        plt.grid()

        ##### Tercer canal Aceleracion filtrado (dividido por 9.81)
        ax2_2 = fig2.add_subplot(313, sharex=ax0_2)
        varo = orientacion(st[i2].get_id())
        ax2_2.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax2_2.transAxes,color='k', fontsize=10)
        max2g = float(max2) / 100.0
        
        if abs(max2g) < 0.01:
            max2gt = format(max2g, '.4f')
        else:
            max2gt = format(max2g, '.2f')
        
        ax2_2.text(0.81, 0.95,'PGA: '+str(max2gt)+' m/s\u00B2',horizontalalignment='left',verticalalignment='top',transform = ax2_2.transAxes)
        sy2_2 = filt2.data  # Ya esta en metros
        plt.plot(t2, sy2_2, 'r', linewidth=ancho)
        plt.xlabel('Tiempo [s]')
        ax2_2.text(xfil, -0.16, msg_fil, verticalalignment='top', horizontalalignment='left', color='k', fontsize=10, transform=ax2_2.transAxes)
        plt.grid()

        # Generar archivo Aceleracion (multiplicado por 100)
        tmpfile2 = fileout1 + "RED_" + network + "_" + estacion + "_" + location + "_ACCm.png"
        fig2.savefig(tmpfile2, format='png', bbox_inches='tight')
        os.chmod(tmpfile2, 0o0777)
        ######################################################################################################################

        ############################################## Para unidad de mg: x 0.981 ##############################################
        ##### Primer canal Aceleracion filtrado (multiplicado por 1000/981)
        fig2 = plt.figure(figsize=(10, 8))
        ax0_2 = fig2.add_subplot(311)
        ax0_2.set_title('[ACELERACION] '+network+'.'+estacion+'.'+location+' / '+str(titulo_hora))
        varo = orientacion(st[i0].get_id())
        ax0_2.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax0_2.transAxes,color='k', fontsize=10)
        max0g = float(max0) * (1000.0/981.0)
        
        if abs(max0g) < 0.01:
            max0gt = format(max0g, '.4f')
        else:
            max0gt = format(max0g, '.2f')
        
        ax0_2.text(0.81, 0.95,'PGA: '+str(max0gt)+' mg',horizontalalignment='left',verticalalignment='top',transform = ax0_2.transAxes)
        sy0_2 = filt0.data * 100 * (1000.0/981.0)  # Multiplicar por 100
        plt.plot(t0, sy0_2, 'b', linewidth=ancho)
        plt.grid()

        ##### Segundo canal Aceleracion filtrado (multiplicado por 0.981)
        ax1_2 = fig2.add_subplot(312, sharex=ax0_2)
        varo = orientacion(st[i1].get_id())
        ax1_2.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax1_2.transAxes,color='k', fontsize=10)
        max1g = float(max1) * (1000.0/981.0)
        
        if abs(max1g) < 0.01:
            max1gt = format(max1g, '.4f')
        else:
            max1gt = format(max1g, '.2f')
        
        ax1_2.text(0.81, 0.95,'PGA: '+str(max1gt)+' mg',horizontalalignment='left',verticalalignment='top',transform = ax1_2.transAxes)
        sy1_2 = filt1.data * 100 * (1000.0/981.0)  # Multiplicar por 100
        plt.plot(t1, sy1_2, 'g', linewidth=ancho)
        plt.ylabel('Aceleracion [mg]')
        plt.grid()

        ##### Tercer canal Aceleracion filtrado (multiplicado por 0.981)
        ax2_2 = fig2.add_subplot(313, sharex=ax0_2)
        varo = orientacion(st[i2].get_id())
        ax2_2.text(0.01, 0.95,varo,verticalalignment='top', horizontalalignment='left',transform=ax2_2.transAxes,color='k', fontsize=10)
        max2g = float(max2) * (1000.0/981.0)
        
        if abs(max2g) < 0.01:
            max2gt = format(max2g, '.4f')
        else:
            max2gt = format(max2g, '.2f')
        
        ax2_2.text(0.81, 0.95,'PGA: '+str(max2gt)+' mg',horizontalalignment='left',verticalalignment='top',transform = ax2_2.transAxes)
        sy2_2 = filt2.data * 100 * (1000.0/981.0)  # Multiplicar por 100
        plt.plot(t2, sy2_2, 'r', linewidth=ancho)
        plt.xlabel('Tiempo [s]')
        ax2_2.text(xfil, -0.16, msg_fil, verticalalignment='top', horizontalalignment='left', color='k', fontsize=10, transform=ax2_2.transAxes)
        plt.grid()

        # Generar archivo Aceleracion (multiplicado por 100)
        tmpfile2 = fileout1 + "RED_" + network + "_" + estacion + "_" + location + "_ACCmg.png"
        fig2.savefig(tmpfile2, format='png', bbox_inches='tight')
        os.chmod(tmpfile2, 0o0777)
        ######################################################################################################################
        
        if ProcessSpeed:
            #Velocidad channel 01
            figv = plt.figure(figsize=(10, 8))
            ax0v = figv.add_subplot(311)
            ax0v.set_title('[VELOCIDAD] '+network+'.'+estacion+'.'+location+' / '+str(titulo_hora))
            vy0 = filt0.copy()
            vy0 = vy0.integrate(method='cumtrapz')
            vy0v = 100*vy0.data
            varo = orientacion(st[i0].get_id())
            ax0v.text(0.01, 0.95, varo,verticalalignment='top', horizontalalignment='left',transform=ax0v.transAxes,color='k', fontsize=10)
            max0v = round(vy0v.max(),4)
            max0nv = round(vy0v.min(),4)
            if abs(max0nv) > max0v:
                max0v = max0nv
            if abs(max0v) < 0.01:
                max0v = format(max0v, '.4f')
            else:
                max0v = format(max0v, '.2f')
            pgv[0] = max0v
            ax0v.text(0.81, 0.95,'PGV: '+str(max0v)+' cm/s',horizontalalignment='left',verticalalignment='top',transform = ax0v.transAxes)
            plt.plot(t0, vy0v,'b',linewidth=ancho)
            plt.grid()

            #Velocidad channel 02
            ax1v = figv.add_subplot(312, sharex=ax0v)
            vy1 = filt1.copy()
            vy1 = vy1.integrate(method='cumtrapz')
            vy1v = 100*vy1.data
            varo = orientacion(st[i1].get_id())
            ax1v.text(0.01, 0.95, varo,verticalalignment='top', horizontalalignment='left',transform=ax1v.transAxes,color='k', fontsize=10)
            max1v = round(vy1v.max(),4)
            max1nv = round(vy1v.min(),4)
            if abs(max1nv) > max1v:
                max1v = max1nv
            if abs(max1v) < 0.01:
                max1v = format(max1v, '.4f')
            else:
                max1v = format(max1v, '.2f')
            pgv[1] = max1v
            ax1v.text(0.81, 0.95,'PGV: '+str(max1v)+' cm/s',horizontalalignment='left',verticalalignment='top',transform = ax1v.transAxes)
            plt.plot(t1, vy1v,'g',linewidth=ancho)
            plt.ylabel('Velocidad [cm/s]')
            plt.grid()
                
            #Velocidad channel 03
            ax2v = figv.add_subplot(313, sharex=ax0v)
            vy2 = filt2.copy()
            vy2 = vy2.integrate(method='cumtrapz')
            vy2v = 100*vy2.data
            varo = orientacion(st[i2].get_id())
            ax2v.text(0.01, 0.95, varo,verticalalignment='top', horizontalalignment='left',transform=ax2v.transAxes,color='k', fontsize=10)
            max2v = round(vy2v.max(),4)
            max2nv = round(vy2v.min(),4)
            if abs(max2nv) > max2v:
                max2v = max2nv
            if abs(max2v) < 0.01:
                max2v = format(max2v, '.4f')
            else:
                max2v = format(max2v, '.2f')
            pgv[2] = max2v
            ax2v.text(0.81, 0.95,'PGV: '+str(max2v)+' cm/s',horizontalalignment='left',verticalalignment='top',transform = ax2v.transAxes)
            plt.xlabel('Time [s]')
            plt.plot(t2, vy2v,'r',linewidth=ancho)
            plt.grid()
            
            #Genera archivos Velocidad
            tmpfile = fileout1+"RED_"+network+"_"+estacion+"_"+location+"_VEL.png"
            figv.savefig(tmpfile, format='png',bbox_inches='tight')
            os.chmod( tmpfile , 0o0777)
            
        if ProcessDisplacement:
            #Desplazamiento channel 01
            figd = plt.figure(figsize=(10, 8))
            ax0d = figd.add_subplot(311)
            ax0d.set_title('[DESPLAZAMIENTO] '+network+'.'+estacion+'.'+location+' / '+str(titulo_hora))
            dy0 = vy0.copy()
            dy0 = dy0.integrate(method='cumtrapz')
            dy0v = 100*dy0.data
            varo = orientacion(st[i0].get_id())
            ax0d.text(0.01, 0.95, varo,verticalalignment='top', horizontalalignment='left',transform=ax0d.transAxes,color='k', fontsize=10)
            max0d = round(dy0v.max(),4)
            max0nd = round(dy0v.min(),4)
            if abs(max0nd) > max0d:
                max0d = max0nd
            if abs(max0d) < 0.01:
                max0d = format(max0d, '.4f')
            else:
                max0d = format(max0d, '.2f')
            pgd[0] = max0d
            ax0d.text(0.81, 0.95,'PGD: '+str(max0d)+' cm',horizontalalignment='left',verticalalignment='top',transform = ax0d.transAxes)
            plt.plot(t0, dy0v,'b',linewidth=ancho)
            plt.grid()
    
            #Desplazamiento channel 02
            ax1d = figd.add_subplot(312, sharex=ax0d)
            dy1 = vy1.copy()
            dy1 = dy1.integrate(method='cumtrapz')
            dy1v = 100*dy1.data
            varo = orientacion(st[i1].get_id())
            ax1d.text(0.01, 0.95, varo,verticalalignment='top', horizontalalignment='left',transform=ax1d.transAxes,color='k', fontsize=10)
            max1d = round(dy1v.max(),4)
            max1nd = round(dy1v.min(),4)
            if abs(max1nd) > max1d:
                max1d = max1nd
            if abs(max1d) < 0.01:
                max1d = format(max1d, '.4f')
            else:
                max1d = format(max1d, '.2f')
            pgd[1] = max1d
            ax1d.text(0.81, 0.95,'PGD: '+str(max1d)+' cm',horizontalalignment='left',verticalalignment='top',transform = ax1d.transAxes)
            plt.plot(t1, dy1v,'g',linewidth=ancho)
            plt.ylabel('Desplazamiento [cm]')
            plt.grid()
                
            #Desplazamiento channel 03
            ax2d = figd.add_subplot(313, sharex=ax0d)
            dy2 = vy2.copy()
            dy2 = dy2.integrate(method='cumtrapz')
            dy2v = 100*dy2.data
            varo = orientacion(st[i2].get_id())
            ax2d.text(0.01, 0.95, varo,verticalalignment='top', horizontalalignment='left',transform=ax2d.transAxes,color='k', fontsize=10)
            max2d = round(dy2v.max(),4)
            max2nd = round(dy2v.min(),4)
            if abs(max2nd) > max2d:
                max2d = max2nd
            if abs(max2d) < 0.01:
                max2d = format(max2d, '.4f')
            else:
                max2d = format(max2d, '.2f')
            pgd[2] = max2d
            ax2d.text(0.81, 0.95,'PGD: '+str(max2d)+' cm',horizontalalignment='left',verticalalignment='top',transform = ax2d.transAxes)
            plt.xlabel('Time [s]')
            plt.plot(t2, dy2v,'r',linewidth=ancho)
            plt.grid()
            
            #Genera archivos Velocidad
            tmpfile = fileout1+"RED_"+network+"_"+estacion+"_"+location+"_DIS.png"
            figd.savefig(tmpfile, format='png',bbox_inches='tight')
            os.chmod( tmpfile , 0o0777)

        if ProcessFourier:
            fy0 = filt0.data
            fy0 = 100*fy0
            fy1 = filt1.data
            fy1 = 100*fy1
            fy2 = filt2.data
            fy2 = 100*fy2

            N  = fy0.size
            T  = 1.0/fps 
            band = 1.0/(2.0*T)
            
            yf1 = np.fft.fft(fy0)
            yf2 = np.fft.fft(fy1)
            yf3 = np.fft.fft(fy2)

            yo1 = np.abs(yf1[0:int(N/2)])
            yo2 = np.abs(yf2[0:int(N/2)])
            yo3 = np.abs(yf3[0:int(N/2)])

            xf = np.linspace(0.0, band, int(N/2))
            #xf = np.linspace(0.01, band, N/2)

            smoothed_signal1 = convolve(yo1, Box1DKernel(80))
            smoothed_signal2 = convolve(yo2, Box1DKernel(80))
            smoothed_signal3 = convolve(yo3, Box1DKernel(80))

            figf = plt.figure(figsize=(10, 8))
            axf = figf.add_subplot(111)
            axf.set_title('[FOURIER] '+network+'.'+estacion+'.'+location)
            varo0 = orientacion(st[i0].get_id())
            varo1 = orientacion(st[i1].get_id())
            varo2 = orientacion(st[i2].get_id())
            axf.text(0.015, 0.98, varo0,verticalalignment='top', horizontalalignment='left',transform=axf.transAxes,color='b', fontsize=10)
            axf.text(0.015, 0.95, varo1,verticalalignment='top', horizontalalignment='left',transform=axf.transAxes,color='g', fontsize=10)
            axf.text(0.015, 0.92, varo2,verticalalignment='top', horizontalalignment='left',transform=axf.transAxes,color='r', fontsize=10)

            #axf.loglog(xf,   (2.0/N)*smoothed_signal1, 'b',basex=10,linewidth=1)
            #axf.loglog(xf,   (2.0/N)*smoothed_signal2, 'g',basex=10,linewidth=1)
            #axf.loglog(xf,   (2.0/N)*smoothed_signal3, 'r',basex=10,linewidth=1)
            axf.plot(xf, (2.0/N)*smoothed_signal1, 'b', linewidth=1)
            axf.plot(xf, (2.0/N)*smoothed_signal2, 'g', linewidth=1)
            axf.plot(xf, (2.0/N)*smoothed_signal3, 'r', linewidth=1)
            axf.set_xscale('log')  # Establece la escala logarítmica en el eje x
            
            axf.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))

            plt.ylabel('Amplitud de fourier [cm/s2*s]')
            plt.xlabel('Frecuencia [Hz]')
            plt.grid()

            #Genera archivos de Fourier
            tmpfile = fileout1+"RED_"+network+"_"+estacion+"_"+location+"_FOU.png"
            figf.savefig(tmpfile, format='png',bbox_inches='tight')
            os.chmod( tmpfile , 0o0777)

        #nombre de canales para datos sin filtrar
        canal1 = st[i0].stats.channel
        canal2 = st[i1].stats.channel
        canal3 = st[i2].stats.channel

        #nombre de los canales para datos filtrados
        canal1f = canal1+"f"
        canal2f = canal2+"f"
        canal3f = canal3+"f"

        #guarda la aceleracion los maximos sin filtro
        MyAcc[random][network][estacion][location].update({canal1:pgaf[0]})
        MyAcc[random][network][estacion][location].update({canal2:pgaf[1]})
        MyAcc[random][network][estacion][location].update({canal3:pgaf[2]})

        #guarda la aceleracion los maximos con filtro
        MyAcc[random][network][estacion][location].update({canal1f:pga[0]})
        MyAcc[random][network][estacion][location].update({canal2f:pga[1]})
        MyAcc[random][network][estacion][location].update({canal3f:pga[2]})

        plt.close("all")

        #Escritura de archivos
        array_de_duplas = [(1, 'cm/s2',''), (1.0/980.0, 'g','g'), (1.0/100.0, 'm/s2','m'), (1000.0/980.0, 'mg','mg')]

        # Recorrer el array de duplas
        for dupla in array_de_duplas:
            # Acceder a los elementos de la dupla
            factor = float(dupla[0])
            unidad = dupla[1]
            textex = dupla[2]
            print("Estoy en dupla: "+str(textex))

            nsinf0 = sinf0*factor
            nsinf1 = sinf1*factor
            nsinf2 = sinf2*factor
            print("pga[0]: "+str(pga[0]))
            print("factor: "+str(factor))
            npga0  = float(pga[0])*factor
            npga1  = float(pga[1])*factor
            npga2  = float(pga[2])*factor
            npga0f = float(pgaf[0])*factor # Sin filtro
            npga1f = float(pgaf[1])*factor # Sin filtro
            npga2f = float(pgaf[2])*factor # Sin filtro

            #Seccion para cuardar con en MyAccPlus con diferentes secciones.
            if textex == "":
                ntextex = 'gal'
            else:
                ntextex = textex
            MyAccPlus[random][network][estacion][location].update({ntextex:{}})

            #guarda la aceleracion los maximos sin filtro
            MyAccPlus[random][network][estacion][location][ntextex].update({canal1:npga0f})
            MyAccPlus[random][network][estacion][location][ntextex].update({canal2:npga1f})
            MyAccPlus[random][network][estacion][location][ntextex].update({canal3:npga2f})

            vector_sf = math.sqrt(npga0f**2 + npga1f**2 + npga2f**2)
            MyAccPlus[random][network][estacion][location][ntextex].update({"vsf":vector_sf})

            #guarda la aceleracion los maximos con filtro
            MyAccPlus[random][network][estacion][location][ntextex].update({canal1f:npga0})
            MyAccPlus[random][network][estacion][location][ntextex].update({canal2f:npga1})
            MyAccPlus[random][network][estacion][location][ntextex].update({canal3f:npga2})

            vector_cf = math.sqrt(npga0**2 + npga1**2 + npga2**2)
            MyAccPlus[random][network][estacion][location][ntextex].update({"vcf":vector_cf})

            ffile = fileout1+"RED_"+network+"_"+estacion+"_"+location+textex+".txt"
            f = codecs.open(ffile, "w", "utf-8")
            f.write("# CENTRO DE MONITOREO DE ESTACIONES SISMICAS\n")
            f.write("# \n")
            f.write("# 1. INFORMACION SOBRE LA ESTACION SISMICA\n")
            f.write("# GRUPO: %s\n" % (network))
            f.write("# ESTACION: %s (%s)\n" % (estacion,nombre_estacion))
            f.write("# CANALES: %s %s %s\n" % (canal1,canal2,canal3))
            f.write("# FRECUENCIA DE MUESTREO (Hz): %s\n" % (fps))
            f.write("# COORDENADAS: %s,%s\n" % (xlat,xlon))
            f.write("# \n")
            f.write("# 2. INFORMACION SOBRE EL SISMO\n")
            f.write("# FECHA: %s\n" % (sismo['fecha']))
            f.write("# HORA INICIO (Local): %s\n" % (sismo['hora']))
            f.write("# LATITUD: %s\n" % (sismo['latitud']))
            f.write("# LONGITUD: %s\n" % (sismo['longitud']))
            f.write("# PROFUNDIDAD (km): %s\n" % (sismo['profundidad']))
            f.write("# MAGNITUD: %s\n" % (sismo['magnitud']))
            f.write("# FUENTE: %s\n" % (sismo['fuente']))
            f.write("# \n")
            f.write("# 3. INFORMACION SOBRE EL REGISTRO\n")
            f.write("# HORA INICIO (UTC-0): %s\n" % (str(UTCDateTime(tr0.stats.starttime))))
            f.write("# NUMERO DE DATOS: %d\n" % (tr0.stats.npts))
            f.write("# UNIDAD: %s\n" % (unidad))
            f.write("# ACELERACIONES MAXIMAS: %f %f %f\n" % (npga0f,npga1f,npga2f))
            #f.write("# ACELEROGRAFO: %s\n" % (nombre_equ))
            f.write("# \n")
            f.write("# 4. COMENTARIOS\n")
            f.write("# CORREGIDO POR LINEA BASE\n")
            f.write("# \n")
            f.write("# 5. DATOS DEL REGISTRO\n")
            f.write("# %s %s %s\n" % (canal1,canal2,canal3))
    
            stoto = [nsinf0,nsinf1,nsinf2]
            ftoto = list(map(list, zip(*stoto)))
            
            np.savetxt(f, ftoto, fmt="%16.8f")
            f.close()
            os.chmod( ffile , 0o0777)

        uestacion   = estacion
    return MyAcc

#Funcion que pre procesa para espectro
def premake_spectrum(random1):
    global alat,alon,frecbajo,frecalto,esta,ancho,fileout1,utc,filtro,StreamFil,ProcessSpectrum,fileout

    n = int(st.count()/3)
    fileout1 = fileout+str(random1)+"/"
    s = 0
    for x in range(0, n):
        i0 = s          #0,3,6
        i1 = s + 1      #1,4,7
        i2 = s + 2      #2,5,8...
        s  = i2 + 1

        #Frecuencia de muestreo
        fps = st[i0].stats.sampling_rate
        #Juntar coordenadas de estaciones

        estacion= st[i0].stats.station
        network = st[i0].stats.network
        location= st[i0].stats.location
        channel = st[i0].stats.channel

        if ProcessSpectrum:
            accels0 = (100.0/980.0)*StreamFil[i0].data
            accels1 = (100.0/980.0)*StreamFil[i1].data
            accels2 = (100.0/980.0)*StreamFil[i2].data

            T = 1.0/fps
            escalax = np.logspace(-2, 1, 100)   # EScala x para todos
            osc_freqs = 1.0/escalax             # en Periodos para todos

            make_spectrum(T,accels0,accels1,accels2,osc_freqs,osc_damping,network,estacion,escalax,fileout1,i0,location)
            #xk = Thread(target=make_spectrum, args=(T,accels0,accels1,accels2,osc_freqs,osc_damping,network,estacion,escalax,fileout1))
            #xk.start()
        
        logs ("Espectro de la estacion: "+str(estacion)+" terminada correctamente.")

#Funcion que define la orientacion en texto legible
def orientacion(txt):
    var = txt[-1]
    direccion = ''
    if var == 'Z':
        direccion = "UD"
    if var == 'N':
        direccion = "NS"
    if var == 'E':
        direccion = "EO"
    direccion = txt[:-3]+direccion
    return direccion

#Funcion que crea la grafia de espectro en segundo plano
def make_spectrum(T,accels0,accels1,accels2,osc_freqs,osc_damping,network,estacion,escalax,fileout1,uu,location):
    global fullpath
    Mycsvfile = fullpath+"/espectros/"+network+".csv"
    existe = os.path.exists(Mycsvfile)
    if(existe):
        logs ("Se encontro archivo de espectros para "+network+" ...")
        datacsv = pd.read_csv(Mycsvfile)
        Myx   = datacsv['x']
        Myest = datacsv['est']
        Myinf = datacsv['inf']
        Mysup = datacsv['sup']
        xnew = np.linspace(Myx.min(), Myx.max(), 300)

        Myf1 = interp1d(Myx, Myest, kind='cubic')
        Myf2 = interp1d(Myx, Myinf, kind='cubic')
        Myf3 = interp1d(Myx, Mysup, kind='cubic')
    else:
        logs ("No existe archivo de espectros, haciendo general...")

    resp_spec0 = pyrotd.calc_spec_accels(T, accels0, osc_freqs, osc_damping)
    resp_spec1 = pyrotd.calc_spec_accels(T, accels1, osc_freqs, osc_damping)
    resp_spec2 = pyrotd.calc_spec_accels(T, accels2, osc_freqs, osc_damping)

    figs = plt.figure(figsize=(10, 8))
    axs = figs.add_subplot(111)
    axs.set_title('[ESPECTRO DE RESPUESTA] '+network+':'+estacion)
    varo1 = orientacion(st[uu].get_id())
    varo2 = orientacion(st[uu+1].get_id())
    varo3 = orientacion(st[uu+2].get_id())
    axs.text(0.015, 0.98, varo1,verticalalignment='top', horizontalalignment='left',transform=axs.transAxes,color='b', fontsize=10)
    axs.text(0.015, 0.95, varo2,verticalalignment='top', horizontalalignment='left',transform=axs.transAxes,color='g', fontsize=10)
    axs.text(0.015, 0.92, varo3,verticalalignment='top', horizontalalignment='left',transform=axs.transAxes,color='r', fontsize=10)

    axs.plot(escalax, resp_spec0.spec_accel, 'b', linewidth=1)
    axs.plot(escalax, resp_spec1.spec_accel, 'g', linewidth=1)
    axs.plot(escalax, resp_spec2.spec_accel, 'r', linewidth=1)

    if(existe):
        cc = 255.0
        mc1 = (127/cc,127/cc,127/cc)
        mc2 = (255/cc,124/cc,128/cc)
        mc3 = (204/cc,153/cc,0/cc)
        axs.plot(xnew, Myf1(xnew),color=mc1, linewidth=1)
        axs.plot(xnew, Myf2(xnew),color=mc2, linewidth=1, linestyle='--')
        axs.plot(xnew, Myf3(xnew),color=mc3, linewidth=1, linestyle='--')

        axs.text(0.015, 0.89, "Grupo: "+str(network) ,verticalalignment='top', horizontalalignment='left',transform=axs.transAxes,color=mc1, fontsize=10)
        axs.text(0.015, 0.86, "Limite superior (Tr=1000)" ,verticalalignment='top', horizontalalignment='left',transform=axs.transAxes,color=mc3, fontsize=10)
        axs.text(0.015, 0.83, "Limite inferior (Tr=1000)" ,verticalalignment='top', horizontalalignment='left',transform=axs.transAxes,color=mc2, fontsize=10)

    axs.set( xlabel='Periodo (Seg)', xscale='log', ylabel='5%-Amortiguamiento Aceleracion espectral (g)')
    axs.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y,pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(y),0)))).format(y)))
    plt.grid()

    #Genera archivos de espectro
    tmpfile = fileout1+"RED_"+network+"_"+estacion+"_"+location+"_SPE.png"
    figs.savefig(tmpfile, format='png',bbox_inches='tight')
    os.chmod( tmpfile , 0o0777)

#Funcion que escrib en log
def logs(mensaje):
    global fullpath
    fechayhora = str(datetime.datetime.now())
    with open(fullpath+"/py/proceso.log", "a") as myfile:
        myfile.write("\n"+fechayhora[0:19]+" - "+mensaje)
    print(str(mensaje))

#Funcion que inserta valores maximos de un evento sismico
def SetGaltoSql(MyAcc,nId):
    global MyAccPlus
    print("MyAcc")
    print(str(MyAccPlus))
    #print(str(MyAcc))
    #import mysql.connector
    #connection = mysql.connector.connect(host='localhost', database='joomla', user='sysop', password='sysop',auth_plugin='mysql_native_password')
    connection = connection_db()
    logs ("Conectando al servidor local")

    texto = list(MyAcc.keys())[0]

    if connection:
        #cur = conn.cursor()
        cursor = connection.cursor()
        n_redes = len(MyAcc[texto])
        for j in range(0,n_redes):
            myred = list(MyAcc[texto].keys())[j]
            n_estaciones = len(MyAcc[texto][myred])
            for i in range(0,n_estaciones):
                station = list(MyAcc[texto][myred].keys())[i]
                n_location = len(MyAcc[texto][myred][station])
                for k in range(0,n_location):
                    location = list(MyAcc[texto][myred][station].keys())[k]
                    key00   = list(MyAcc[texto][myred][station][location].keys())[0]
                    key01   = list(MyAcc[texto][myred][station][location].keys())[1]
                    key02   = list(MyAcc[texto][myred][station][location].keys())[2]
                    key03   = list(MyAcc[texto][myred][station][location].keys())[3]
                    key04   = list(MyAcc[texto][myred][station][location].keys())[4]
                    key05   = list(MyAcc[texto][myred][station][location].keys())[5]

                    gal00   = MyAcc[texto][myred][station][location][key00]
                    gal01   = MyAcc[texto][myred][station][location][key01]
                    gal02   = MyAcc[texto][myred][station][location][key02]
                    gal03   = MyAcc[texto][myred][station][location][key03]
                    gal04   = MyAcc[texto][myred][station][location][key04]
                    gal05   = MyAcc[texto][myred][station][location][key05]

                    #En gal
                    gal00 = MyAccPlus[texto][myred][station][location]["gal"][key00]
                    gal01 = MyAccPlus[texto][myred][station][location]["gal"][key01]
                    gal02 = MyAccPlus[texto][myred][station][location]["gal"][key02]
                    gal03 = MyAccPlus[texto][myred][station][location]["gal"][key03]
                    gal04 = MyAccPlus[texto][myred][station][location]["gal"][key04]
                    gal05 = MyAccPlus[texto][myred][station][location]["gal"][key05]

                    #En g
                    gal00_g = MyAccPlus[texto][myred][station][location]["g"][key00]
                    gal01_g = MyAccPlus[texto][myred][station][location]["g"][key01]
                    gal02_g = MyAccPlus[texto][myred][station][location]["g"][key02]
                    gal03_g = MyAccPlus[texto][myred][station][location]["g"][key03]
                    gal04_g = MyAccPlus[texto][myred][station][location]["g"][key04]
                    gal05_g = MyAccPlus[texto][myred][station][location]["g"][key05]

                    #En m
                    gal00_m = MyAccPlus[texto][myred][station][location]["m"][key00]
                    gal01_m = MyAccPlus[texto][myred][station][location]["m"][key01]
                    gal02_m = MyAccPlus[texto][myred][station][location]["m"][key02]
                    gal03_m = MyAccPlus[texto][myred][station][location]["m"][key03]
                    gal04_m = MyAccPlus[texto][myred][station][location]["m"][key04]
                    gal05_m = MyAccPlus[texto][myred][station][location]["m"][key05]

                    #En mg
                    gal00_mg = MyAccPlus[texto][myred][station][location]["mg"][key00]
                    gal01_mg = MyAccPlus[texto][myred][station][location]["mg"][key01]
                    gal02_mg = MyAccPlus[texto][myred][station][location]["mg"][key02]
                    gal03_mg = MyAccPlus[texto][myred][station][location]["mg"][key03]
                    gal04_mg = MyAccPlus[texto][myred][station][location]["mg"][key04]
                    gal05_mg = MyAccPlus[texto][myred][station][location]["mg"][key05]

                    sql1 = "INSERT INTO maxgal(code,station,channel,gal,g,m,mg,network,location,myid) VALUES('"+texto+"','"+station+"','"+key00+"','"+str(gal00)+"','"+str(gal00_g)+"','"+str(gal00_m)+"','"+str(gal00_mg)+"','"+str(myred)+"','"+str(location)+"','"+str(nId)+"');"
                    #cur.execute(sql1)
                    cursor.execute(sql1)
                    sql2 = "INSERT INTO maxgal(code,station,channel,gal,g,m,mg,network,location,myid) VALUES('"+texto+"','"+station+"','"+key01+"','"+str(gal01)+"','"+str(gal01_g)+"','"+str(gal01_m)+"','"+str(gal01_mg)+"','"+str(myred)+"','"+str(location)+"','"+str(nId)+"');"
                    #cur.execute(sql2)
                    cursor.execute(sql2)
                    sql3 = "INSERT INTO maxgal(code,station,channel,gal,g,m,mg,network,location,myid) VALUES('"+texto+"','"+station+"','"+key02+"','"+str(gal02)+"','"+str(gal02_g)+"','"+str(gal02_m)+"','"+str(gal02_mg)+"','"+str(myred)+"','"+str(location)+"','"+str(nId)+"');"
                    #cur.execute(sql3)
                    cursor.execute(sql3)
                    sql4 = "INSERT INTO maxgal(code,station,channel,gal,g,m,mg,network,location,myid) VALUES('"+texto+"','"+station+"','"+key03+"','"+str(gal03)+"','"+str(gal03_g)+"','"+str(gal03_m)+"','"+str(gal03_mg)+"','"+str(myred)+"','"+str(location)+"','"+str(nId)+"');"
                    #cur.execute(sql4)
                    cursor.execute(sql4)
                    sql5 = "INSERT INTO maxgal(code,station,channel,gal,g,m,mg,network,location,myid) VALUES('"+texto+"','"+station+"','"+key04+"','"+str(gal04)+"','"+str(gal04_g)+"','"+str(gal04_m)+"','"+str(gal04_mg)+"','"+str(myred)+"','"+str(location)+"','"+str(nId)+"');"
                    #cur.execute(sql5)
                    cursor.execute(sql5)
                    sql6 = "INSERT INTO maxgal(code,station,channel,gal,g,m,mg,network,location,myid) VALUES('"+texto+"','"+station+"','"+key05+"','"+str(gal05)+"','"+str(gal05_g)+"','"+str(gal05_m)+"','"+str(gal05_mg)+"','"+str(myred)+"','"+str(location)+"','"+str(nId)+"');"
                    #cur.execute(sql6)
                    cursor.execute(sql6) 
                    #print(str(station))
        connection.commit()
        cursor.close()
        connection.close()
        #conn.commit()
        #cur.close()
        #conn.close()
    else:
        logs ("No hay conexion al servidor. Verifique.")

def putRunTime():
    global random,fullpath
    ahora = str(time.time())
    f = open(fullpath+"/tmp/"+str(random)+".pid","w")
    f.write(ahora)
    f.close()

def getRunTime():
    global random,fullpath
    t = 0
    try:
        f = open(fullpath+"/tmp/"+str(random)+".pid","r")
        t = f.read()
        f.close()
    except:
        t = 0
    return t

def verify_launch(cmd):
    limitesg = 20
    sp = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    
    # Obtener la salida estándar y de error
    salida_estandar, salida_error = sp.communicate()
    if sp.returncode != 0:
        logs("Error: {}".format(sp.returncode))
        logs( "Salida de error:")
        logs( str(salida_error))
    else:
        logs( "Salida estándar:")
        logs( str(salida_estandar))

    ejec = False
    logs("Se lanzo sub proceso ("+str(sp)+"), se verificara 3 veces dentro de 10 sg...")
    #putRunTime()
    for k in range(0,3):
        logs('Esperando 10 sg para verificacion '+str(k)+'...'+str(cmd))
        time.sleep(10)
        #vf = sp.poll()
        ahora = time.time()
        mpid = getRunTime()
        if mpid == '' or mpid == 0:
            mpid = 0
        logs("mpid: "+str(mpid))
        pid = float(mpid)
        df = ahora - pid
        logs("df: "+str(df))
        if df > limitesg:
            logs('NO se esta ejecutando, intentando lanzarlo de nuevo')
            sp = subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            logs(str(sp))
        else:
            ejec = True
            break
    if ejec:
        logs("Todo se ejecuto normal, OK")
    return sp

#Funcion principal del programa principal
if __name__ == "__main__":
    #Lectura de variables
    fullp = os.path.dirname(os.path.abspath(__file__))
    fullp = '/'.join(fullp.split("/")[:-1])+"/.env"
    #fullp = "/var/www/html/sensor"
    load_dotenv(fullp)
    web_server      = os.getenv('WEB_SERVER')
    seedlink_port   = os.getenv('SEEDLINK_PORT')
    server_ip       = os.getenv('SERVER_IP')
    fdsn_port       = os.getenv('FDSN_PORT')
    tiempoespera    = os.getenv('TIEMPOESPERA')
    path_seiscomp   = os.getenv('PATH_SEISCOMP')
    fullpath        = os.getenv('FULL_PATH')
    prefijo         = os.getenv('PREFIJO')

    fileout  = fullpath+"/events/"
    random = random_key(10)
    logs ("Random creado es: "+random+"\n")
    logs("se creo la carpeta: "+random)
    Evento = GetArg(sys.argv)
    
    utc_time = datetime.datetime.strptime(Evento['FechayHora'], "%Y-%m-%dT%H:%M:%SZ")
    tota_second = MyTiempo - (datetime.datetime.now() - utc_time).total_seconds()

    if tota_second > 0:
        tota_second = tota_second + 5
        logs("Se lanzo muy temprano, esperando "+str(tota_second)+" segundos para ejecutar...")
        time.sleep(tota_second)
    
    print (str(Evento))
    #exit()

    logs ("Extrayendo inventario...\n")
    hini = datetime.datetime.now()    
    Inventario = GetInventory()

    logs("Se va trabajar en modo: "+Evento['Modo'])
    if Evento['Modo'] == 's':
        logs ("Conexion a SeedLink\n")
        st = GetStreamSeedLink(Evento)
    else:
        logs ("Conexion a Fdsnws\n")
        st = GetStreamArcLink(Evento)
    
    #print(str(st))
    st.merge(fill_value='latest')

    st.sort()
    n = int(st.count()/3)
    logs ("Antes de uniformizar se tiene: "+str(n)+" estaciones para procesar")
    
    if n != 0:
        stn1 = st.copy()
        #print(str(stn1))
        stn2 = solo3channels(stn1)
        n = int(stn2.count()/3)
        logs ("Se seleccionaron solo 3 canales por estacion dando "+str(n)+" estaciones por procesar")

        stg = uniformizo_streams(n,Evento,stn2)
        logs ("Uniformizando Stream Stg: "+str(stg.count()/3))

        stg.attach_response(Inventario)
        logs ("Stg Attach response: "+str(stg.count()/3))
        logs(str(stg))

        stg.remove_sensitivity()
        logs ("Stg Remove Sensitivity: "+str(stg.count()/3))

        n = int(stg.count()/3)
        logs ("Se confirma que se tienen: "+str(n)+" estaciones\n")
        st = stg.copy()
        print(str(st))
        #nnn = st.count()
        #for yyy in range(0,nnn):
        #    ttt = st[yyy].stats.channel
        #    print(str(ttt))
        #if Evento["NewCode"] == "N":

        MyAcc = FileProcess(Evento,Inventario)
        nId = Evento["Id"]
        #print( str(Evento))
        if Evento['Share']:
            # Obtener la hora actual en formato UTC con ObsPy
            hora_actual_obspy = UTCDateTime.now()
            # Restar 5 horas
            hora_actual_ajustada_obspy = hora_actual_obspy - 5 * 3600  # 5 horas en segundos
            # Convertir a cadena en formato ISO (opcional)
            Evento['hora_impresion'] = hora_actual_ajustada_obspy.isoformat()
            resultado_dict = {"evento": Evento, "aceleraciones": MyAccPlus}
            resultado_json = json.dumps(resultado_dict, ensure_ascii=False, indent=2)
            # Escribir el resultado en un archivo
            with open(fullpath+'/py/evento.json', 'w') as archivo:
                archivo.write(resultado_json)
                logs("Se ha creado el archivo 'evento.json'")

        SetGaltoSql(MyAcc,nId)

        hfin = datetime.datetime.now()
        lapso = hfin - hini
        logs ("Se demoro: " + str(lapso) + " segundos en proceso inicial")
        
        # Ojo esta linea que sigue se comenta para no hacer los espectros de disenho y test mas rapido
        if Evento['Respuesta']:
            logs ("Iniciando procesamiento de espectros...")
            premake_spectrum(random) 
            logs ("termino procesamiento de espectros")

        #genera_pdf(random)

        hfin = datetime.datetime.now()
        lapso = hfin - hini
        logs ("Se demoro: " + str(lapso) + " segundos en proceso final")
        logs ("Modo de deteccion: "+str(sismo['sourceigp']))
        ftigp = str(sismo['sourceigp'])
        plantilla = Evento['Plantilla']

        if Evento['NewPDF']:
            if Evento['NewEmail']:
                care = 1
            else:
                care = 0
            mycmd = ['/usr/bin/php', fullpath+'/php/fork_report.php',str(random),str(care),str(ftigp),plantilla,'0']
            process = verify_launch(mycmd)
            #process = subprocess.Popen(mycmd, stdout=subprocess.PIPE)
            logs (str(process))

            # Disparo de script envio_notifica_whatsapp.py
            logs("Inicio de envío de notifiaciones por whatsapp - envio_notifica_whatsapp.py")
            # envio_script_path = os.path.join(os.getenv('FULL_PATH'), 'py/envio_notifica_whatsapp.py')
            envio_script_path = '/var/www/html/servicios_python_audas/sistema_monitoreo_sensores/modules/notificacion_sismos/envio_notifica_whatsapp.py'

            # CAptura el resultado
            result = subprocess.run(['python3', envio_script_path], capture_output=True, text=True)
            logs(result.stdout)
            if result.returncode != 0:
                logs("Error al ejecutar envio_notifica_whatsapp.py: " + result.stderr)
                
        
        if Evento['NewEmail'] and not Evento['NewPDF']:
            mycmd = ['/usr/bin/php', fullpath+'/php/fork_email.php',str(random),str(ftigp),'0']
            #process = subprocess.Popen(mycmd, stdout=subprocess.PIPE)
            process = verify_launch(mycmd)
            logs ("Se lazo fork PHP para correo de evento: "+str(mycmd))
            logs (str(process))
    else:
        logs ("No hay registros, se procede a actualizar el evento y nada mas")
        sismo = GetParametros(Evento,random)

    #Rutinas para generar mapas de localizacion del epicentro con estaciones.
