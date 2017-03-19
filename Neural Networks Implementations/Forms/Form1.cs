using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace Neural_Networks_Implementations
{
    public partial class Form1 : Form
    {
        public DataReader Data;
        public TheModel Model;
        TheTester ModelTester;
        public int FeatureOnX;
        public int FeatureOnY;
        public int bias ;
        public double eta;
        public int epochs;
        public int class1;
        public int class2;
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            Data = new DataReader();

            comboBox1.Items.Add("Feature 1");
            comboBox1.Items.Add("Feature 2");
            comboBox1.Items.Add("Feature 3");
            comboBox1.Items.Add("Feature 4");

            comboBox2.Items.Add("Feature 1");
            comboBox2.Items.Add("Feature 2");
            comboBox2.Items.Add("Feature 3");
            comboBox2.Items.Add("Feature 4");

            comboBox3.Items.Add("1- setosa");
            comboBox3.Items.Add("2- versicolor");
            comboBox3.Items.Add("3- virginica");

            comboBox4.Items.Add("1- setosa");
            comboBox4.Items.Add("2- versicolor");
            comboBox4.Items.Add("3- virginica");

        }

        private void DrawDiscretePoints(object sender, EventArgs e)
        {
            chart1.Series[0].Points.Clear();
            chart1.Series[1].Points.Clear();
            chart1.Series[2].Points.Clear();
            FeatureOnX = comboBox1.Text[8] - '0' - 1;
            FeatureOnY = comboBox2.Text[8] - '0' - 1;

            for (int i = 0; i < Data.NumberOfClasses; i++)
            {
                for (int j = 0; j < Data.NumberOfTrainingInstances; j++)
                {
                    chart1.Series[i].Points.AddXY(Data.TrainingData[i, j, FeatureOnX], Data.TrainingData[i, j, FeatureOnY]);
                }
                for (int j = 0; j < Data.NumberOfTestingInstances; j++)
                {
                    chart1.Series[i].Points.AddXY(Data.TestingData[i, j, FeatureOnX], Data.TestingData[i, j, FeatureOnY]);
                }
            }
         
        }

        private void Train(object sender, EventArgs e)
        {
            int.TryParse(textBox2.Text, out epochs);
            double.TryParse(textBox1.Text, out eta);
            class1 = comboBox3.Text[0] - '0' - 1;
            class2 = comboBox4.Text[0] - '0' - 1;
            if (checkBox1.Checked == true)
                bias = 1;
            else
                bias = 0;
            MessageBox.Show("Training session started");
            Model = new TheModel(this);
            MessageBox.Show("Training session ended successfully");
        }

        private void Tester(object sender, EventArgs e)
        {
            MessageBox.Show("Start testing the model");
            ModelTester = new TheTester(this);
        }

    }
}
