using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace LinearniRegrese
{
    public class ReadInfo
    {
        public void readInf(MLContext mlcon, string dataPath = "ceny_domu.csv")
        {

            IDataView dataView = mlcon.Data.LoadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');
            var split = mlcon.Data.TrainTestSplit(dataView, testFraction: 0.3);

            var pipeline = mlcon.Transforms.CopyColumns("Label", "Cena").
                Append(mlcon.Transforms.Concatenate("Features", "Kvalita", "Plocha", "RokVýstavby", "RokProdeje")).
                Append(mlcon.Regression.Trainers.Sdca());

            var model = pipeline.Fit(split.TrainSet);

            var predictBaddie = model.Transform(split.TestSet);

            var metrics = mlcon.Regression.Evaluate(predictBaddie);

            Console.WriteLine($"prumerna absolutni odchylka: {metrics.MeanAbsoluteError} \n" +
                $"prumerna kvadraticka chyba: {metrics.MeanSquaredError}");
            predictionEngineMine(mlcon, model);
        }
        public void predictionEngineMine(MLContext mlcon, 
            Microsoft.ML.Data.TransformerChain<Microsoft.ML.Data.RegressionPredictionTransformer<Microsoft.ML.Trainers.LinearRegressionModelParameters>> model)
        {
            var predictionModel = mlcon.Model.CreatePredictionEngine<HouseData, Predictions>(model);

            var newHouse = new HouseData { Kvalita = 4, Plocha = 60, RokVýstavby = 2000, RokProdeje = 2026 };
            var newPrediction = predictionModel.Predict(newHouse);

            Console.WriteLine($"predpovezena cena: {newPrediction.Score}");
        }
    }
}
