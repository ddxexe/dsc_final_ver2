import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.patches 
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw, ImageFont


def zoom_img(img,zoom):
    width, height = img.size
    width2 = int(width * zoom)
    height2 = int(height * zoom)
    img2 = img.resize((width2,height2))

    num_pixels =  math.ceil(800/zoom) ** 2
    num_values = 3 * num_pixels 
    st.write("There are " + str(num_pixels) + " pixels on-screen, meaning " + str(num_values) + " values to keep track of.")
    #draw.text((10,10),img2_txt,fill='white',size=30)

    return img2

def crop_img(img,val):
    return img.crop((0,0,math.ceil(800/val),math.ceil(800/val)))


def pixels_demo():
    st.title("Visualizing Results")

    st.write("Our dataset consists solely of images. Ironically, this poses some challenges from a data visualization standpoint. \
         The method my colleagues and I used was by examining the color values of each pixel of the image. Since our images lack \
         an alpha channel, each pixel contains three float values, corresponding to the intensities of the RED, GREEN, and BLUE subpixels ([RED,GREEN,BLUE]). \
         For instance, the color white could be represented with the list [0,0,0], while the color magenta could be represented with the list [1,0,1]. \
         Each image therefore contains a huge volume of information.")
    
    zoom_val = st.slider("Magnification level",min_value = 1,max_value = 5,value = 1,step=1)
    img = Image.open("./sample_img.png")
    cropped_img = crop_img(img,zoom_val)
    zoomed_image = zoom_img(cropped_img,zoom_val)

    zoomed_image_width = zoomed_image.width
    st.image(zoomed_image)

    st.write("Our dataset consists of a series of roughly 1150 of these 800 by 800 images, many of which will be passed into \
             our machine learning model several times. All in all it means that a relatively short training routine \
             can take over 20 minutes to run. We begin the training process with a pre-determined set of weights and biases \
             (think the lists I used for subpixels, but much longer).")
    
    saliency_map = Image.open("./saliency_map.png")
    saliency_map_overlay = Image.open("./saliency_map_overlay.png")
    
    st.write("Let's actually set up what any of this looks like. Here's a saliency map, which applies the optimized weights to the actual image.\
             As the name implies, the highlighted regions indicate which parts of the region had the greatest effect on classifying the image.")
    add_line_breaks(1)
    overlay = st.checkbox(label = "Overlay", value = False)

    if overlay == False:
        st.image(saliency_map)
    else:
        st.image(saliency_map_overlay)

    add_line_breaks(1)

    st.write("It visualizes the exact regions I thought it would. The visual identifiers are structural - they pertain to things like \
             the edges of certain structures, and large empty spaces. And the saliency map reflects this well.")
    add_line_breaks(11)



def accuracy_demo():
    st.title("Accuracy isn't everything.")
    st.write("Accuracy is functionally our default way of measuring how good a model is at making predictions. But this isn't \
             always the case. Consider two scenarios of analyzing 2000 tumors:")
    
    st.markdown("**Model A**")
    st.markdown("- 1900 are benign, and 100 are malignant.")
    st.markdown("- 90 percent of all benign images are classified correctly.")
    st.markdown("- 50 percent of all malignant images are classified correctly.")
    
    add_line_breaks(1)

    st.markdown("**Or Model B**")
    st.markdown("- 1900 are benign, and 100 are malignant.")
    st.markdown(" - 50 percent of all benign images are classified correctly.")
    st.markdown("- 90 percent of malignant images are classified correctly.")

    add_line_breaks(1)

    st.markdown("Model A's accuracy is **88 percent**, while Model B's is **52 percent**. However, Model B almost always correctly \
                flags malignant tumors, while Model A's 50 percent accuracy is no better than random chance. Model B is nominally less \
                accurate, but is more effective at saving lives. This does **not** mean that Model B is outright better; by this logic \
                a model which flags every image as malignant is perfect because it identifies every malignant case. There's a balancing \
                act with no set in stone solution.")


    st.write('One way of seeing the interplay between accuracy and sensitivity is with an ROC curve.')
    
    df_roc = pd.read_csv('roc_df.csv')
    fig2 = plt.figure(figsize=(9,4))
    st.line_chart(df_roc,x='False Positive Rate',y='True Positive Rate')
    #s = st.slider('select y',min_value = 0.0,max_value=1.0,step=0.01)
    #plt.plot([0,1],[0.5,0.8])
    #select_x = df_roc.loc[df_roc['False Positive Rate'] == s,'True Positive Rate'].values
    #plt.show()
    #st.write(f'For x = {select_x}, y = {s}')
    add_line_breaks(5)


    fig = plt.figure(figsize=(9,4))
    df = pd.read_csv('predictions.csv')

    sns.color_palette("husl",9)

    sns.set_style("whitegrid")
    #sns.set(rc={'axes.facecolor':'lightblue', 'figure.facecolor':'lightgreen'})
    threshold = st.slider('Select a sensitivity threshold',min_value = 0.0,max_value=1.0,step=0.01)
    df['hue'] = (df['Predicted'] > threshold) == (df['Actual'] > threshold)
    num_positive = (df['hue'] == 1).sum()
    num_negative = (df['hue'] == 0).sum()
    total = num_positive + num_negative
    percent_positive = 100 * num_positive/total
    percent_negative = 100 * num_negative/total
    plt.axvline(x=threshold,color='r')
    sns.stripplot(data = df, x = 'Predicted',y='Category',jitter=True,hue='hue',orient='y',legend=False)
    plt.title("Managing Incorrect Classifications")
    plt.show()
    st.pyplot(fig)
    st.write("We have " + str(round(percent_positive,2)) + " percent of cases which are correctly flagged and " + str(round(percent_negative,2)) + " percent \
             of cases which are incorrectly flagged.")




def comparing_models_demo():
    st.title("Comparing Results")
    st.write("Our group is hardly the only one doing this sort of research. This kind of research was already underway before \
             AI was in vogue. Here are some comparisons between our models. The results are promising but all of these numbers \
             should be taken with an immense grain of salt.")
    
    st.write("A trained pathologist can can correctly identify the subclass of tumor about 95 percent of the time. \
            Our model would need to consistently surpass these accuracy thresholds before we can even humor any sort \
            of practical applications. So let's make some comparisons: ")
    
    df_models = pd.read_csv('comparing_models.csv',encoding='unicode_escape')
    fig2 = plt.figure(figsize=(9,4))
    plt.bar(df_models['Model'], df_models['Accuracy'])
    plt.ylim(np.min(df_models['Accuracy'] - 5),100)
    plt.xlabel("Model (Year | Main Authors)")
    plt.ylabel("Peak Accuracy")
    plt.title("Comparing Model Accuracies")
    st.pyplot(fig2)

    #add_line_breaks(5)
    #fig3 = plt.figure(figsize=(9,4))
    #plt.bar(df_models['Model'], df_models['Sensitivity'])
    #plt.ylim(np.min(df_models['Sensitivity'] - 5),100)
    #plt.xlabel("Model (Year | Main Authors)")
    #plt.ylabel("Peak Sensitivity")
    #plt.title("Comparing Model Sensitivities")
    #st.pyplot(fig2)

    st.write("There are some severe caveats to this. Most of these studies have very small sample studies, making them extremely \
             ill-equipped for any sort of real-world application. For instance, our group's sample size (1151) only covers 21 actual tumors. \
             The limitations of this study are mostly self-evident but several AI fine-tuning models like k-fold cross validation are \
             debatably ineffective for our sample size.")





def print_text_before_fig_2():
    st.title("Comparing models")

    

def add_line_breaks(n):
    for i in range(n):
        st.write("\n")


def main():

    pixels_demo()

    
    accuracy_demo()

    comparing_models_demo()


if __name__ == "__main__":
    main()