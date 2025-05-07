

using LinearniRegrese;
using Microsoft.ML;

var mlContax = new MLContext(seed:0);
ReadInfo ri = new ReadInfo();
ri.readInf(mlContax);
