
using Microsoft.ML.Data;

namespace LinearniRegrese
{
    public class HouseData
    {
        [LoadColumn(0)]public float Kvalita;
        [LoadColumn(1)] public float Plocha;
        [LoadColumn(2)] public float RokVýstavby;
        [LoadColumn(3)] public float RokProdeje;
        [LoadColumn(4)] public float Cena; //label





    }
}
