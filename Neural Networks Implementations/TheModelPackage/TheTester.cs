using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Networks_Implementations
{
    class TheTester
    {
        double v;
        double y;
        double Accuracy;
        int[,] ConfusionMatrix;
        public TheTester(Form1 TheMainForm)
        {
            ConfusionMatrix = new int[2, 2];
            Test(TheMainForm);
            Print();
        }
        private void Test(Form1 TheMainForm)
        {
            
            for(int i=0;i<TheMainForm.Data.NumberOfTestingInstances;i++)
            {
                //class 1 instanc testing
                v = TheMainForm.Model.Weights[0]* TheMainForm.bias + TheMainForm.Data.TestingData[TheMainForm.class1, i, TheMainForm.Model.Feature1] * TheMainForm.Model.Weights[1] +
                        TheMainForm.Data.TestingData[TheMainForm.class1, i, TheMainForm.Model.Feature2] * TheMainForm.Model.Weights[2];
                y = TheMainForm.Model.SignumActivationFunction(v);
                if (y == 1)
                    ConfusionMatrix[0, 0]++;
                else
                    ConfusionMatrix[0, 1]++;

                //class 2 instanc testing
                v = TheMainForm.Model.Weights[0] * TheMainForm.bias + TheMainForm.Data.TestingData[TheMainForm.class2, i, TheMainForm.Model.Feature1] * TheMainForm.Model.Weights[1] +
                       TheMainForm.Data.TestingData[TheMainForm.class2, i, TheMainForm.Model.Feature2] * TheMainForm.Model.Weights[2];
                y = TheMainForm.Model.SignumActivationFunction(v);
                if (y == -1)
                    ConfusionMatrix[1, 1]++;
                else
                    ConfusionMatrix[1, 0]++;
            }
            Accuracy = 100*(ConfusionMatrix[0, 0] + ConfusionMatrix[1, 1]) / (2 * TheMainForm.Data.NumberOfTestingInstances);
           
        }
        private void Print()
        {
            MessageBox.Show("The confusion Matrix is:\n\n" +
               ConfusionMatrix[0, 0] + "\t\t\t" + ConfusionMatrix[0, 1] + "\n" +
               ConfusionMatrix[1, 0] + "\t\t\t" + ConfusionMatrix[1, 1] + "\n\n\n" +
               "The accuracy is:  " + Accuracy.ToString() + "%");
        }
    }
}
