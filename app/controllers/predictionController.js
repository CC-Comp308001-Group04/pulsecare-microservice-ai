const tf = require("@tensorflow/tfjs");
const fs = require("fs");
const path = require("path");

// Load training and testing data
const heartDiseaseData = require("../../heartdisease.json");
const heartDiseaseTestingData = require("../../heartdiseaseTesting.json");

// Function to train and predict
exports.trainAndPredict = async function (req, res) {

  // const trainingData = tf.tensor2d(heartDiseaseData.map(item => [
  //     item.age, item.sex, item.cp, item.trestbps, item.chol,
  //     item.fbs, item.restecg, item.thalach, item.exang, item.oldepeak,
  //     item.slope, parseFloat(item.ca), parseFloat(item.thal),

  // ]));

  //receive user input
  const userInput = req.body;
  const inputData = tf.tensor2d([[
    parseFloat(userInput.age),
    userInput.sex === "1" ? 1 : 0,
    parseFloat(userInput.cp),
    parseFloat(userInput.trestbps),
    parseFloat(userInput.chol),
    userInput.fbs === "1" ? 1 : 0,
    parseFloat(userInput.thalach)
  ]]);

  // Prepare the data
  const trainingData = tf.tensor2d(
    heartDiseaseData.map((item) => [
      parseFloat(item.age),
      item.sex === "1" ? 1 : 0,
      parseFloat(item.cp),
      parseFloat(item.trestbps),
      parseFloat(item.chol),
      item.fbs === "1" ? 1 : 0,
     // parseFloat(item.restecg),
      parseFloat(item.thalach),
     // item.exang === "1" ? 1 : 0,
    //  parseFloat(item.oldepeak),
    //  parseFloat(item.slope),
     // item.ca === "?" ? 0 : parseFloat(item.ca),
     // item.thal === "?" ? 0 : parseFloat(item.thal),
    ])
  );

  const outputData = tf.tensor2d(
    heartDiseaseData.map((item) => [
      //item.num > 0 ? 1 : 0, // presence (1)    absence (0)
      item.num === 0 ? 0 : 1, // Assuming 'num' is the label indicating presence (1) or absence (0) of heart disease
    ])
  );

  // Build and compile the model
  const model = tf.sequential();
  //   model.add(
  //     tf.layers.dense({ inputShape: [13], units: 10, activation: "relu" })
  //   ); // Adjust the number of input units and units
  //   model.add(tf.layers.dense({ units: 1, activation: "sigmoid" })); // Binary classification output layer
  //   model.add(
  //     tf.layers.dense({ inputShape: [13], units: 32, activation: "relu" })
  //   );
  //   model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  //   model.add(tf.layers.dense({ units: 32, activation: "relu" }));
  //   model.add(tf.layers.dense({ units: 1, activation: "sigmoid" })); // Binary classification output layer
  model.add(
    tf.layers.dense({
      inputShape: [7],
      units: 20,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    })
  );
  model.add(tf.layers.dropout(0.5));
  model.add(
    tf.layers.dense({
      units: 10,
      activation: "relu",
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    })
  );
  model.add(tf.layers.dropout(0.5));
  model.add(tf.layers.dense({ units: 5, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
  model.compile({
    optimizer: tf.train.adam(),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  // Train the model
  await model.fit(trainingData, outputData, {
    epochs: 100, // Number of epochs
    validationSplit: 0.2, // Split the data into training (80%) and validation (20%)
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`
        );
      },
    },
  });
  const prediction = model.predict(inputData);
  await prediction.data().then(predictionData => {
    const predictedClass = predictionData[0] > 0.5 ? "Yes" : "No";
    res.json({ prediction: predictedClass });
  });
  // Predict the testing data
  // const testingData = tf.tensor2d(
  //   heartDiseaseTestingData.map((item) => [
  //     item.age,
  //     item.sex,
  //     item.cp,
  //     item.trestbps,
  //     item.chol,
  //     item.fbs,
  //    // item.restecg,
  //     item.thalach,
  //    // item.exang,
  //    // item.oldepeak,
  //    // item.slope,
  //    // parseFloat(item.ca),
  //    // parseFloat(item.thal),
  //   ])
  // );

  // const predictions = model.predict(testingData);
  // predictions.print(); // Optionally print the predictions to the console

  // // Convert predictions to an array and send as a response
  // const predictedClasses = await predictions.dataSync();
  // res.json(predictedClasses.map((pred) => (pred > 0.5 ? 1 : 0))); // Convert sigmoid outputs to binary classes
};
