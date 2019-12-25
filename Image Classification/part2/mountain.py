#!/usr/local/bin/python3
#
# Authors: [Hrishkikesh Paul Ishita Kumar Rushabh Shah hrpaul ishkumar shah12]
#
# Mountain ridge finder
# Based on skeleton code by D. Crandall, Oct 2019
#

from PIL import Image
from numpy import *
from scipy.ndimage import filters
import sys
import imageio

# calculate "Edge strength map" of an image
#
def edge_strength(input_image):
    grayscale = array(input_image.convert('L'))  # convert image to monochrome
    filtered_y = zeros(grayscale.shape)
    filters.sobel(grayscale,0,filtered_y)
    return sqrt(filtered_y**2)

# draw a "line" on an image (actually just plot the given y-coordinates
#  for each x-coordinate)
# - image is the image to draw on
# - y_coordinates is a list, containing the y-coordinates and length equal to the x dimension size
#   of the image
# - color is a (red, green, blue) color triple (e.g. (255, 0, 0) would be pure red
# - thickness is thickness of line in pixels
#
def draw_edge(image, y_coordinates, color, thickness):
    for (x, y) in enumerate(y_coordinates):

        a = int(max(y-(thickness/2), 0))

        b= int(min(y+int(thickness/2), image.size[1]-1 ))
        for t in range(a,b):
            image.putpixel((x, t), color)
    return image


# main program
#
(input_filename, gt_row, gt_col) = ('test_images/'+sys.argv[1],0,0)

# load in image
input_image = Image.open(input_filename)
#print(input_image)
# compute edge strength mask
edge_strength = edge_strength(input_image)
imageio.imwrite('edges.jpg', uint8(255 * edge_strength / (amax(edge_strength))))

# You'll need to add code here to figure out the results! For now,
max_pixel=[]
for i in range(edge_strength.shape[1]):
   maxx=0
   for j in range(edge_strength.shape[0]):
       if edge_strength[j][i] > maxx:
           maxx = edge_strength[j][i]
           row_index = j

   max_pixel.append(row_index)

# just create a horizontal centered line.
ridge1 = [ edge_strength.shape[0]/2 ] * edge_strength.shape[1]

ridge2 = max_pixel

# output answer
#imageio.imwrite("output.jpg", draw_edge(Image.open(input_filename), ridge1, (255, 0, 0), 2))
imageio.imwrite("output simple.jpg", draw_edge(Image.open(input_filename), ridge2, (0, 0, 255), 2))

sum_column = sum(edge_strength,axis=0)
ridge3 = []



def transition_prob(row1, row2):                                  # Trasition probability function
    x = abs(row1 - row2)
    return (edge_strength.shape[0]-x)/edge_strength.shape[0]


emission_prob = zeros((edge_strength.shape[0], edge_strength.shape[1])) # emission prob is a matrix which stores emission for each row
                                                                        # corresponding to a column    

for i in range(edge_strength.shape[1]):                                # precomputing emission prob                                 
    # emission_prob = []

    for j in range(edge_strength.shape[0]):
        ## emission probability

        emission_prob[j][i] = log(edge_strength[j][i] / sum_column[i])

observable = []
for i in range(edge_strength.shape[1]):
    observable.append(i)

probabilities = []

if observable[0] == 0:
    for j in range(edge_strength.shape[0]):
        probabilities.append(emission_prob[j][1])
ridge3.append(where(probabilities == max(probabilities))[0][0])

for i in range(1, edge_strength.shape[1]):                                  # viterbi Algorithm

    for k in range(edge_strength.shape[0]):
        p = []
        for j in range(edge_strength.shape[0]):

            p.append(emission_prob[k][i] + probabilities[j] + log(transition_prob(k, j)))

        probabilities[k] = max(p)

    ridge3.append(where(probabilities == max(probabilities))[0][0])

imageio.imwrite("output map.jpg", draw_edge(input_image, ridge3, (255, 0, 0), 2))
#def valid_index(index):
# s=0
# for j in range(52,58):
#     print(edge_strength[j][6])
#     s+=edge_strength[j][6]


    
initial_row = int(sys.argv[2])
col = int(sys.argv[3])

ridge4 = []
prior=[]
ridge4.append(initial_row)                          ## Bidirectional viterbi for human part
row=initial_row
ridge5 = []
if (col!=0):
    for j in range(edge_strength.shape[0]):

        prior.append(emission_prob[j][col])

    for i in range(col+1, edge_strength.shape[1]):

        for k in range(edge_strength.shape[0]):
            p = []
            for j in range(edge_strength.shape[0]):
                p.append(emission_prob[k][i] + prior[j] + log(transition_prob(k, j)))

            prior[k] = max(p)

            # print(prior[row-2:row+2])
        row = argmax(prior[row-6:row+7]) + row -6
        ridge4.append(row)
    row = initial_row
    prior=[]
    for j in range(edge_strength.shape[0]):
        prior.append(emission_prob[j][col])
    for i in range(col-1, -1,-1):

        for k in range(edge_strength.shape[0]):
            p = []
            for j in range(edge_strength.shape[0]):
                p.append(emission_prob[k][i] + prior[j] + log(transition_prob(k, j)))

            prior[k] = max(p)

            # print(prior[row-2:row+2])
        row = argmax(prior[row-3:row+4]) + row-3
        ridge5.append(row)

    ridge5=ridge5[::-1]
    ridge = ridge5+ridge4

    #print(ridge)
    imageio.imwrite("output_human.jpg", draw_edge(Image.open(input_filename), ridge, (0,255, 0), 3))
if (col==0):
    for j in range(edge_strength.shape[0]):

        prior.append(emission_prob[j][col])

    for i in range(col+1, edge_strength.shape[1]):

        for k in range(edge_strength.shape[0]):
            p = []
            for j in range(edge_strength.shape[0]):
                p.append(emission_prob[k][i] + prior[j] + log(transition_prob(k, j)))

            prior[k] = max(p)

            # print(prior[row-2:row+2])
        row = argmax(prior[row-3:row+4]) + row -3
        ridge4.append(row)
    
    ridge = ridge4

    #print(ridge)
    imageio.imwrite("output_human.jpg", draw_edge(Image.open(input_filename), ridge, (0,255, 0), 3))
    
