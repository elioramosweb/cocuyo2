import matplotlib
import streamlit as st 
#matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import matplotlib.image as mpimg
import matplotlib.cm as cm
from   matplotlib.colorbar import ColorbarBase
from   matplotlib.colors import Normalize
from   matplotlib.pyplot import figure
from   scipy.interpolate import griddata
import pandas as pd
import numpy as np
import io
from io import StringIO
import math



###################################
# funcion para interpolar los datos
###################################

def interpolaDatos(THETA,R,magnitude):

  THETAi  = np.linspace(min(THETA),max(THETA),num=720)

  Ri      = np.linspace(min(R),max(R),num=360)

  theta,r = np.meshgrid(THETAi,Ri,indexing='xy')

  z = griddata((THETA,R),magnitude,(theta,r),method="linear")

  z = np.round(z,2)

  return((THETAi,Ri,z))


#####################################################
# funcion para promediar la medidas a azimuto 0 y 360
#####################################################

def fixMagnitude(magnitude):

  inx0    = np.arange(0,6)

  inx360  = np.arange(len(magnitude) - 6,len(magnitude))

  magnitude0    = magnitude[inx0]

  magnitude360  = magnitude[inx360]

  magnitudePromedio = (magnitude0 + magnitude360)/2

  magnitude[inx0]   = magnitudePromedio

  magnitude[inx360] = magnitudePromedio

  return(magnitude)

##########################################
# funcion para seleccinar tabla de colores
##########################################

def seleccionaColor(miColor):
  if (miColor == "azul"):
    micolor = cm.YlGnBu
  elif (miColor == "anaranjado"):
    micolor = cm.autumn
  elif (miColor == "gris"):
    micolor = cm.Greys
  return(micolor)


#############################################
# funcion para extraer columnas de data frame
#############################################

def extraeColumnas(df):

  azimuth   = np.array(df['azimuth'])
  elevation = np.array(df['elevation'])
  magnitude = np.array(df['magnitude'])

  return((azimuth,elevation,magnitude))


##################################
# función para hacer el mapa polar
################################## 

def polarPlot(df,
              miTitulo,
              miFecha,
              miColor="azul",
              puntosMuestreo = False,
              direccionReloj = False,
              mostrarContorno = False):

  # titulo = "Pitahaya"
  # fecha  = "8/Febrero/2016"

  # extraer columnas del dataframe

  azimuth,elevation,magnitude = extraeColumnas(df)

  # para seleccinar tabla de colores

  micolor = seleccionaColor(miColor)

  fig = pyplot.figure()

  # para promediar la medidas a azimuto 0 y 360

  magnitude = fixMagnitude(magnitude)

  # conversion de angulos a radianes

  theta = np.radians(azimuth)
  phi   = np.radians(elevation)

  phi = abs(phi - math.pi/2.0)/(math.pi/2.0)

  # definicion de eje para barra de colores

  cax = fig.add_axes([0.25, 0.03, 0.5, 0.03])

  # definicion de eje para grafica polar

  ax = fig.add_axes([0.1, 0.15, 0.8, 0.8],
                  frameon=True,
                  polar=True,facecolor='#d5de9c',aspect='equal')
                  #polar=True,axisbg='#d5de9c',aspect='equal')

  if direccionReloj:
    ax.set_theta_direction(-1)
    ax.set_theta_offset(math.pi/2.0)

  labels = np.unique(theta)

  ax.set_xticks(labels[:-1])
  ax.set_yticks(())

  #rCard  = 1.4

  ax.set_xticklabels(['N', '$30^\degree$', '$60^\degree$', 'E',
  '$120^\degree$', '$150^\degree$','S', '$210^\degree$', '$240^\degree$','W', '$300^\degree$','$330^\degree$'],
  ha='center')

  # ax.text(0.0,rCard,"N",fontsize=14,weight='bold')
  # ax.text(math.pi/2,rCard,"E",fontsize=14,weight='bold')
  # ax.text(math.pi,rCard,"S",fontsize=14,weight='bold')
  # ax.text(3*math.pi/2,rCard,"W",fontsize=14,weight='bold')
  #
  thetai,phii,z = interpolaDatos(theta,phi,magnitude)

  vmin = 18.4
  vmax = 21.8

  #vmin = 18.0
  #vmax = 22.0 
  
  ax.set_title(miTitulo + "\n" + str(miFecha))

  ax.contourf(thetai,phii,z,10,cmap=micolor,vmin=vmin,vmax=vmax)

  ##ax.contourf(thetai,phii,z,10,linewidth=2,cmap=cm.YlGnBu_r,vmax=vmax)

  if direccionReloj:
    ang = 0.0
  else:
    ang = math.pi/2

  if mostrarContorno:
    ax.contour(thetai,phii,z,10,colors="k",linewidths=0.5)

  # mostrar puntos de muestreo

  if puntosMuestreo:
    ax.scatter(theta,phi)
    ax.set_rmin(0)

  # especificar radio del mapa polar

  ax.set_rmax(0.99)

  #vmin = min(magnitude)
  #vmax = max(magnitude)

  # rango de magnitudes en la tabla de colores


  # # para mostrar las unidades al lado de la barra de colores

  # if direccionReloj:
  #   ax.text(np.radians(151),1.28,"$mag/arcsec^2$",
  #     weight="bold",
  #     fontsize=16)
  # else:
  #   ax.text(np.radians(299),1.28,"$mag/arcsec^2$",
  #           weight="bold",
  #           fontsize=16)


  cbar = ColorbarBase(cax,orientation='horizontal',cmap=micolor,
               spacing='proportional',
               label="$mag/arcsec^2$",
               norm=Normalize(vmin=vmin, vmax=vmax))

  
 

  ##pyplot.savefig(outfile)

  ##pyplot.close()
  
  ##fig.show()

  return(fig)


st.set_page_config(layout="wide")


###########################
#### PROGRAMA PRINCIPAL 
###########################


############
## TITULO ##
############


st.sidebar.image("cocuyo.png")

st.write("<h1>Visualización de Datos de Brillo Nocturno</h1><br>",unsafe_allow_html=True)


## datos de ejemplo 

usarEjemplo = st.sidebar.checkbox("Utilizar archivo de ejemplo",value=False)

## mostrar los puntos de muestreo 

mostrarPuntos = st.sidebar.checkbox("Mostrar puntos de muestreo",value=False)

## mostrar líneas de contorno 

mostrarContornos = st.sidebar.checkbox("Mostrar contornos",value=False)

## dirección de las manecillas del reloj 

direccion = st.sidebar.checkbox("Gráfico en dirección de las manecillas del reloj",value=False)

## paleta de colores 

colores = st.sidebar.selectbox(
    'Paleta de colores',
    ('azul', 'anaranjado', 'gris'))

st.sidebar.write("<br>",unsafe_allow_html=True)

if usarEjemplo: 
    uploaded_file = "ejemplo.dat"
else:
    uploaded_file = st.sidebar.file_uploader("Seleccionar archivo de datos:")

if not uploaded_file:
    st.image("logo.png")


miTitulo = st.sidebar.text_input("Título de la gráfica",value="Título aquí")

miFecha = st.sidebar.text_input("Fecha de los datos",value="Fecha aquí")


if uploaded_file is not None:
    
    data = np.loadtxt(uploaded_file)

    dataframe = pd.DataFrame(data,
                             columns=['azimuth','elevation','magnitude'])
    
    st.sidebar.table(dataframe.style.format("{:4.1f}"))

   ## miTitulo = "Pitahaya"
    ##miFecha  = "8/Febrero/2016"
    ##outfile = "ejemplo.png"
    ##miColor = "azul"
    
    fig = polarPlot(dataframe,miTitulo,miFecha,colores,
                    mostrarPuntos,direccion,mostrarContornos)

    fn = "mapa" + miTitulo + miFecha + ".png"
    img = io.BytesIO()
    pyplot.savefig(img,format='png',bbox_inches='tight')
    
    btn = st.download_button(
        label ='Bajar Imagen',
        data=img,
        file_name=fn,
        mime="image/png"
    )
    
    st.pyplot(fig)

