const express = require("express");
const bodyParser = require("body-parser");
const predictionRoutes = require("./app/routes/predictionRoutes");

const app = express();

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use("/api", predictionRoutes);

const port = process.env.PORT || 4000;

app.get("/", (req, res) => {
  res.send("Heart Disease Predictor API");
});

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
