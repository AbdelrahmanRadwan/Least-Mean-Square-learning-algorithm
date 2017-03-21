using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Networks_Implementations
{
    public class TheModel
    {
        int epochs;
        int num_input;
        int num_training;
        public double bias;
        public int Class1, Class2; //zero based
        public int Feature1, Feature2; // zero based;
        double eta;
        double mse_thres;
        double v; // Linear Combiner
        double y; // the response
        double MinFeature1, MinFeature2, MaxFeature1, MaxFeature2;
        double[] error; //vector 1 X number of training
        public double[] Weights; // vector of  ( number of inputs +1 ) x 1
        double[] OldWeights; // vector of  ( number of inputs +1 ) x 1
        double[] MeanSquareError; // vector of 1 X  epochs
        int k;

        public TheModel(Form1 TheMainForm)
        {
            this.epochs = TheMainForm.epochs;
            this.num_input = 2;
            this.num_training = TheMainForm.Data.NumberOfTrainingInstances * this.num_input;
            this.bias = TheMainForm.bias;
            this.eta = TheMainForm.eta;
            this.mse_thres = TheMainForm.mse_thres;
            this.Class1 = TheMainForm.class1;
            this.Class2 = TheMainForm.class2;
            this.Feature1 = TheMainForm.FeatureOnX;
            this.Feature2 = TheMainForm.FeatureOnY;
            this.MinFeature1 = this.MinFeature2 = 1000000;
            this.MaxFeature1 = this.MaxFeature2 = -1000000;
            this.k = 0;
            error = new double[num_training];
            Weights = new double[num_input + 1];
            OldWeights = new double[num_input + 1];
            MeanSquareError = new double[epochs];

            for (int i = 0; i < num_input + 1; i++)

                Weights[i] = 0.5;
            //PreProcessing(TheMainForm);

            PerceptronTrainer(TheMainForm);
        }
        private void PreProcessing(Form1 TheMainForm)
        {
            for (int i = 0; i < TheMainForm.Data.NumberOfTrainingInstances; i++)
            {
                MinFeature1 = Math.Min(MinFeature1, TheMainForm.Data.TrainingData[Class1, i, Feature1]);
                MinFeature1 = Math.Min(MinFeature1, TheMainForm.Data.TrainingData[Class2, i, Feature1]);

                MinFeature2 = Math.Min(MinFeature2, TheMainForm.Data.TrainingData[Class1, i, Feature2]);
                MinFeature2 = Math.Min(MinFeature2, TheMainForm.Data.TrainingData[Class2, i, Feature2]);

                MaxFeature1 = Math.Max(MaxFeature1, TheMainForm.Data.TrainingData[Class1, i, Feature1]);
                MaxFeature1 = Math.Max(MaxFeature1, TheMainForm.Data.TrainingData[Class2, i, Feature1]);

                MaxFeature2 = Math.Max(MaxFeature2, TheMainForm.Data.TrainingData[Class1, i, Feature2]);
                MaxFeature2 = Math.Max(MaxFeature2, TheMainForm.Data.TrainingData[Class2, i, Feature2]);
            }

            for (int i = 0; i < TheMainForm.Data.NumberOfTrainingInstances; i++)
            {
                TheMainForm.Data.TrainingData[Class1, i, Feature1] = 2 * (TheMainForm.Data.TrainingData[Class1, i, Feature1] - MinFeature1) / (MaxFeature1 - MinFeature1) - 1;
                TheMainForm.Data.TrainingData[Class2, i, Feature1] = 2 * (TheMainForm.Data.TrainingData[Class2, i, Feature1] - MinFeature1) / (MaxFeature1 - MinFeature1) - 1;
                TheMainForm.Data.TrainingData[Class1, i, Feature2] = 2 * (TheMainForm.Data.TrainingData[Class1, i, Feature2] - MinFeature2) / (MaxFeature2 - MinFeature2) - 1;
                TheMainForm.Data.TrainingData[Class2, i, Feature2] = 2 * (TheMainForm.Data.TrainingData[Class2, i, Feature2] - MinFeature2) / (MaxFeature2 - MinFeature2) - 1;
            }
        }
        private void DeepCopyWeights()
        {
            for (int i = 0; i < num_input + 1; i++)
                OldWeights[i] = Weights[i];
        }

        public int SignumActivationFunction(double v)
        {
            if (v >= 0)
                return 1;
            else
                return -1;
        }
        private double CalculateMeanSquareError()
        {
            double result = 0;
            for (int i = 0; i < num_training; i++)
                result = result + error[i] * error[i];

            return 0.5 * result / num_training;
        }

        bool Equality()
        {
            for(int i=0;i<num_input + 1;i++)
            {
                if (Weights[i] - OldWeights[i] >= 0.1)
                    return false;
            }
            return true;
        }
        private void PerceptronTrainer(Form1 TheMainForm)
        {
            for (int i = 0; i < epochs; i++)
            {
                OldWeights = Weights;
                k = 0;
                // class 1 process
                for (int j = 0; j < TheMainForm.Data.NumberOfTrainingInstances; j++)
                {
                    if (j % 2 == 0)
                    {

                        v = Weights[0]*bias + TheMainForm.Data.TrainingData[Class1, j, Feature1] * Weights[1] +
                            TheMainForm.Data.TrainingData[Class1, j, Feature2] * Weights[2];
                        y = v;//  Linear activation function 
                        error[k] = 1 - y; //1 is the first class

                        Weights[0] = (Weights[0] + (eta * error[k] * 1)) * bias;
                        Weights[1] = Weights[1] + (eta * error[k] * TheMainForm.Data.TrainingData[Class1, j, Feature1]);
                        Weights[2] = Weights[2] + (eta * error[k] * TheMainForm.Data.TrainingData[Class1, j, Feature2]);
                        k++;

                        v = Weights[0]*bias + TheMainForm.Data.TrainingData[Class2, j, Feature1] * Weights[1] +
                            TheMainForm.Data.TrainingData[Class2, j, Feature2] * Weights[2];
                        y = v;//  Linear activation function 
                        error[k] = -1 - y; // 2 is the first class

                        Weights[0] = (Weights[0] + (eta * error[k] * 1)) * bias;
                        Weights[1] = Weights[1] + (eta * error[k] * TheMainForm.Data.TrainingData[Class2, j, Feature1]);
                        Weights[2] = Weights[2] + (eta * error[k] * TheMainForm.Data.TrainingData[Class2, j, Feature2]);
                        k++;

                    }
                    else
                    {

                        v = Weights[0]*bias + TheMainForm.Data.TrainingData[Class2, j, Feature1] * Weights[1] +
                            TheMainForm.Data.TrainingData[Class2, j, Feature2] * Weights[2];
                        y = v;//  Linear activation function 
                        error[k] = 1 - y; // 1 is the first class

                        Weights[0] = (Weights[0] + (eta * error[k] * 1)) * bias;
                        Weights[1] = Weights[1] + (eta * error[k] * TheMainForm.Data.TrainingData[Class2, j, Feature1]);
                        Weights[2] = Weights[2] + (eta * error[k] * TheMainForm.Data.TrainingData[Class2, j, Feature2]);
                        k++;

                        v = Weights[0]*bias + TheMainForm.Data.TrainingData[Class1, j, Feature1] * Weights[1] +
                                TheMainForm.Data.TrainingData[Class1, j, Feature2] * Weights[2];
                        y = v;//  Linear activation function 
                        error[k] = -1 - y; // 2 is the second class

                        Weights[0] = (Weights[0] + (eta * error[k] * 1)) * bias;
                        Weights[1] = Weights[1] + (eta * error[k] * TheMainForm.Data.TrainingData[Class1, j, Feature1]);
                        Weights[2] = Weights[2] + (eta * error[k] * TheMainForm.Data.TrainingData[Class1, j, Feature2]);
                        k++;
                    }
                    /*if(Equality())
                    {
                        break;
                    }*/
                    
                }
                MeanSquareError[i] = CalculateMeanSquareError();
                 if (MeanSquareError[i] < mse_thres)
                     break;
            }

        }
    }
}
