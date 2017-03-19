using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace Neural_Networks_Implementations
{
    public class DataReader
    {
        public double[, ,] TrainingData;
        public double[, ,] TestingData;
        int NumberOfFeatures;
        public int NumberOfClasses;
        public int NumberOfTrainingInstances;
        public int NumberOfTestingInstances;
        int NumberOfInstances;
        string[] Lines;
        string[] Tokens;

        public DataReader()
        {
            NumberOfClasses = 3;
            NumberOfFeatures = 4;
            NumberOfInstances = 50;
            NumberOfTrainingInstances = 30;
            NumberOfTestingInstances = 20;

            TrainingData = new double[NumberOfClasses, NumberOfTrainingInstances, NumberOfFeatures];
            TestingData = new double[NumberOfClasses, NumberOfTestingInstances, NumberOfFeatures];

            Lines = System.IO.File.ReadAllLines(@"Iris Data.txt");

            Reader();
        }
        private void Reader()
        {
            for (int i = 0; i<NumberOfClasses;i++)
            {
                //fill the train
                for(int j=0;j<NumberOfTrainingInstances;j++)
                {
                    Tokens = Lines[NumberOfInstances*i+j].Split(',');
                    for (int k = 0; k < NumberOfFeatures; k++)                       
                        double.TryParse(Tokens[k],out TrainingData[i, j, k]);
                }
                //fill the test
                for (int j = 0; j < NumberOfTestingInstances; j++)
                {
                    Tokens = Lines[NumberOfInstances * i + j].Split(',');
                    for (int k = 0; k < NumberOfFeatures; k++)
                        double.TryParse(Tokens[k], out TestingData[i, j, k]);
                }
            }
        }
    }
}
