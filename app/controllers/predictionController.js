

const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const path = require("path");

// Load training and testing data
const heartDiseaseData = require("../../heartdisease.json");

function calculateMean(data, key) {
  return data.reduce((acc, curr) => acc + parseFloat(curr[key]), 0) / data.length;
}

function calculateStdDev(data, key, mean) {
  return Math.sqrt(data.reduce((acc, curr) => acc + Math.pow(parseFloat(curr[key]) - mean, 2), 0) / data.length);
}

// Generate statistics for normalization
function generateStats(data, featureKeys) {
  return featureKeys.reduce((acc, key) => {
    const mean = calculateMean(data, key);
    const stdDev = calculateStdDev(data, key, mean);
    acc[key] = { mean, stdDev };
    return acc;
  }, {});
}

function normalize(data, stats, featureKeys) {
  return data.map(item => {
    return featureKeys.reduce((acc, key) => {
      acc[key] = (parseFloat(item[key]) - stats[key].mean) / stats[key].stdDev;
      return acc;
    }, {});
  });
}

const featureKeys = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'thalach'];
const stats = generateStats(heartDiseaseData, featureKeys);
const normalizedHeartDiseaseData = normalize(heartDiseaseData, stats, featureKeys);

exports.trainAndPredict = async function (req, res) {
  const userInput = req.body;
  const normalizedInput = featureKeys.map(key => (parseFloat(userInput[key]) - stats[key].mean) / stats[key].stdDev);

  const inputData = tf.tensor2d([normalizedInput]);
  const trainingData = tf.tensor2d(normalizedHeartDiseaseData.map(item => featureKeys.map(key => item[key])));
  const outputData = tf.tensor2d(heartDiseaseData.map(item => [item.num === 0 ? 0 : 1]));

  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [featureKeys.length],
    units: 20,
    activation: "relu",
    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
  }));
  model.add(tf.layers.dropout(0.5));
  model.add(tf.layers.dense({ units: 10, activation: "relu", kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }) }));
  model.add(tf.layers.dropout(0.5));
  model.add(tf.layers.dense({ units: 5, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({ optimizer: tf.train.adam(), loss: "binaryCrossentropy", metrics: ["accuracy"] });

  await model.fit(trainingData, outputData, {
    epochs: 100,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`)
    }
  });

  const prediction = model.predict(inputData);
  await prediction.data().then(predictionData => {
    const predictedProbability = predictionData[0] * 100;
    res.json({ probability: `${predictedProbability.toFixed(2)}%` });
  });
};