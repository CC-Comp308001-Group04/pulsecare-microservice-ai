// predictionRoutes.js
const express = require('express');
const router = express.Router();
const predictionController = require('../controllers/predictionController');

router.post('/predict', predictionController.trainAndPredict);

module.exports = router;


// Define the routes module' method
// module.exports = function (app) {

//     app.get('/', function (req, res) {
//         res.render('index', {
//             info: "see the results in console window"
//         })
//     });

//     app.get('/run', index.trainAndPredict);
//     app.post('/input-data', index.trainAndPredict);

// };