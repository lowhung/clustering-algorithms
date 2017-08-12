import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

import utils
from kmeans import Kmeans
from kmedians import Kmedians
from quantize_image import ImageQuantizer
from sklearn.cluster import DBSCAN
import linear_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1', '1.1', '1.2', '1.3.1', '1.3.2', '1.3.4', '1.4', '2.1', '2.2', '4', '4.1', '4.3'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1':
        X = utils.load_dataset('clusterData')['X']

        model = Kmeans(k=4)
        model.fit(X)
        utils.plot_2dclustering(X, model.predict(X))

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    if question == '1.1':
        X = utils.load_dataset('clusterData')['X']
        N, D = X.shape
        # model = Kmeans(k=4)
        # model.fit(X)
        # model.predict(X)
        # objective_value = model.error(X)

        model_array = []
        error_array = []

        for i in range(0,50):
            model = Kmeans(k=4)
            model_array.append(model)
            model_array[i].fit(X)
            model_array[i].predict(X)
            error_array.append(model_array[i].error(X))

        y = min(error_array)
        index = error_array.index(y)
        model = model_array[index]
        y = model.predict(X)
        utils.plot_2dclustering(X,y)

    if question == '1.2':
        X = utils.load_dataset('clusterData')['X']
        N, D = X.shape

        k_values = range(1,11)
        error_array = np.zeros((50, 10))
        y = np.zeros(10)

        for j in k_values:
            model_array = []
            for i in range(0, 50):
                model = Kmeans(k=j)
                model_array.append(model)
                model_array[i].fit(X)
                model_array[i].predict(X)
                error_array[i,j-1] = model_array[i].error(X)
            y[j-1] = np.min(error_array[:,j-1])

        plt.plot(k_values, y, label="Minimum error against k")
        plt.xlabel("k values")
        plt.ylabel("Minimum error")
        plt.legend()
        fname = os.path.join("..", "figs", "q12minerror.pdf")
        plt.savefig(fname)

    if question == '1.3.1':
        X = utils.load_dataset('clusterData2')['X']

        model_array = []
        error_array = []
        for i in range(0,50):
            model = Kmeans(k=4)
            model_array.append(model)
            model_array[i].fit(X)
            model_array[i].predict(X)
            error_array.append(model_array[i].error(X))
            print(error_array)

        y = np.min(error_array)
        index = error_array.index(y)
        model = model_array[index]
        y = model.predict(X)

        utils.plot_2dclustering(X,y)

    if question == '1.3.2':
        X = utils.load_dataset('clusterData2')['X']

        k_values = range(1,11)
        error_array = np.zeros((50, 10))
        y = np.zeros(10)

        for j in k_values:
            model_array = []
            for i in range(0, 50):
                model = Kmeans(k=j)
                model_array.append(model)
                model_array[i].fit(X)
                model_array[i].predict(X)
                error_array[i,j-1] = model_array[i].error(X)
                print(error_array)
            y[j-1] = min(error_array[:,j-1])

        plt.title("Elbow method plot for clusterData2")
        plt.plot(k_values, y, label="K-Means minimum error plot against k values")
        plt.xlabel("k values")
        plt.ylabel("Minimum error")
        plt.legend()
        fname = os.path.join("..", "figs", "q13minerror.pdf")
        plt.savefig(fname)

    if question == '1.3.4':
        X = utils.load_dataset('clusterData2')['X']

        model_array = []
        error_array = []
        for i in range(0,50):
            model = Kmedians(k=4)
            model_array.append(model)
            model_array[i].fit(X)
            model_array[i].predict(X)
            error_array.append(model_array[i].error(X))

        y = min(error_array)
        index = error_array.index(y)
        model = model_array[index]
        y = model.predict(X)

        utils.plot_2dclustering(X,y)

    if question == '1.4':
        X = utils.load_dataset('clusterData2')['X']

        model = DBSCAN(eps=15, min_samples=2)
        y = model.fit_predict(X)

        utils.plot_2dclustering(X,y)
        fname = os.path.join("..", "figs", "clusterdata_dbscan.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)


    if question == '2.1':
        img = utils.load_dataset('dog')['I']/255
        q_image = ImageQuantizer(b=4)
        new_image_1 = q_image.quantize_image(img)

    if question == '2.2':
        img = utils.load_dataset('dog')['I']/255
        q_image = ImageQuantizer(b=1)
        new_image_1 = q_image.dequantize_image(img)

        q_image = ImageQuantizer(b=2)
        new_image_2 = q_image.dequantize_image(img)

        q_image = ImageQuantizer(b=4)
        new_image_4 = q_image.dequantize_image(img)

        q_image = ImageQuantizer(b=6)
        new_image_6 = q_image.dequantize_image(img)

        plt.axis("off")
        plt.figure(1)
        plt.imshow(new_image_1)
        plt.show()
        plt.figure(2)
        plt.imshow(new_image_2)
        plt.show()
        plt.figure(3)
        plt.imshow(new_image_4)
        plt.show()
        plt.figure(4)
        plt.imshow(new_image_6)
        plt.show()

    elif question == "4":
            # loads the data in the form of dictionary
            data = utils.load_dataset("outliersData")
            X = data['X']
            y = data['y']

            # Plot data
            plt.figure()
            plt.plot(X,y,'b.',label = "Training data")
            plt.title("Training data")

            # Fit least-squares estimator
            model = linear_model.LeastSquares()
            model.fit(X,y)
            print(model.w)

            # Draw model prediction
            Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
            yhat = model.predict(Xsample)
            plt.plot(Xsample,yhat,'g-', label = "Least squares fit", linewidth=4)
            plt.legend()
            figname = os.path.join("..","figs","least_squares_outliers.pdf")
            print("Saving", figname)
            plt.savefig(figname)

    elif question == "4.1":
            data = utils.load_dataset("outliersData")
            X = data['X']
            y = data['y']
            z_1 = np.ones((400,1))
            z_2 = np.full((100,1), 0.1)
            z = np.append(z_1, z_2)
            z = z * np.identity(500)

            # Plot data
            plt.figure()
            plt.plot(X,y,'b.',label = "Training data")
            plt.title("Training data")

            # Fit weighted-least-squares estimator
            model = linear_model.WeightedLeastSquares()
            model.fit(X,y,z)
            # print(model.w)

            Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
            yhat = model.predict(Xsample)
            plt.plot(Xsample,yhat,'g-', label = "Least squares fit", linewidth=2)
            plt.legend()
            figname = os.path.join("..","figs","least_squares_outliers.pdf")
            print("Saving", figname)
            plt.savefig(figname)


    elif question == "4.3":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        # print(model.w)

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Draw model prediction
        Xsample = np.linspace(np.min(X), np.max(X), 1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample, yhat, 'g-', label = "Least squares fit", linewidth=4)
        plt.legend()
        figname = os.path.join("..","figs","gradient_descent_model.pdf")
        print("Saving", figname)
        plt.savefig(figname)
