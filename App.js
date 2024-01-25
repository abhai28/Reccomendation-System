const fs = require("fs");

//Class UserBased: Handles the user-based collaborative filtering recommendation system
class UserBased {
  // Constructor initializes the class with data and algorithm parameters
  constructor(data, neighborhoodSize, similarityThreshold, absoluteSimilarity) {
    this.data = data;
    this.neighborhoodSize = neighborhoodSize;
    this.similarityThreshold = similarityThreshold;
    this.totalNeighbours = 0;
    this.noValidNeighbors = 0;
    this.rGreaterFive = 0;
    this.rLessThanOne = 0;
    this.MAE = 0;
    this.predictionsMade = 0;
    this.absoluteSimilarity = absoluteSimilarity;
  }

  // Computes the average rating for each user
  preComputeAverages(ratings) {
    // For each user, compute the sum of all ratings and the number of ratings
    return ratings.map((userRatings) => {
      let sum = 0;
      let count = 0;
      userRatings.forEach((rating) => {
        if (rating !== 0) {
          sum += parseFloat(rating);
          count++;
        }
      });
      return { sum: sum, count: count, avg: count === 0 ? 0 : sum / count };
    });
  }

  // Creates a matrix of size n x m
  createMatrix(n, m) {
    return Array.from({ length: n }, () => Array(m).fill(0));
  }

  // Computes the similarity between two users
  computeSimilarities(ratings, userAverages, N, M, a) {
    //Initializing an array to store the similarity between user a and all other users
    let userSimilarities = Array(N).fill(0);

    //Computing the similarity between user a and all other users
    for (let b = 0; b < N; b++) {
      if (b != a) {
        let similarity = this.pcc(a, b, ratings, userAverages);
        userSimilarities[b] = similarity;
      }
    }

    return userSimilarities;
  }

  //Computes the Pearson Correlation Coefficient between two users
  pcc(a, b, ratings, userAverages) {
    let numerator = 0;
    let denominatorA = 0;
    let denominatorB = 0;

    //Averages of user a and user b
    let r_aavg = userAverages[a].avg;
    let r_bavg = userAverages[b].avg;

    //Computing the numerator and the denominator of the PCC
    for (let i = 0; i < ratings[a].length; i++) {
      let r_ai = ratings[a][i];
      let r_bi = ratings[b][i];
      if (r_ai !== 0 && r_bi !== 0) {
        let diffA = r_ai - r_aavg;
        let diffB = r_bi - r_bavg;
        numerator += diffA * diffB;
        denominatorA += diffA * diffA;
        denominatorB += diffB * diffB;
      }
    }

    if (denominatorA === 0 || denominatorB === 0) return 0;
    //Computing the PCC
    let sim = numerator / (Math.sqrt(denominatorA) * Math.sqrt(denominatorB));
    return sim;
  }

  //Sorts the array by the specified column
  sortByColumn(arr, column) {
    return [...arr].sort((a, b) => b[column] - a[column]);
  }

  //Computes the predicted rating for user a and item p
  prediction(a, p, ratings, neighbourhoodSize, similarities, userAverages) {
    let r_aavg = userAverages[a].avg;
    let numerator = 0;
    let denominator = 0;

    //Identifying the neighbours of user a based on the similarity
    let neighbors = [];
    for (let b = 0; b < ratings.length; b++) {
      let tempSim = similarities[b];
      if (this.absoluteSimilarity) {
        similarities[b] = Math.abs(similarities[b]);
      }
      if (
        ratings[b][p] !== 0 &&
        similarities[b] > this.similarityThreshold &&
        a !== b
      ) {
        neighbors.push([tempSim, ratings[b][p], userAverages[b].avg]);
      }
      similarities[b] = tempSim;
    }

    //If there are no valid neighbours, return the average rating of user a
    if (neighbors.length === 0) {
      this.noValidNeighbors++;
      return r_aavg;
    }
    //If the number of neighbours is less than the specified neighbourhood size, use all the neighbours
    if (neighbors.length < neighbourhoodSize)
      neighbourhoodSize = neighbors.length;

    this.totalNeighbours += neighbourhoodSize;
    //Sorting the neighbours by similarity from highest to lowest
    let topNeighbors = this.sortByColumn(neighbors, 0).slice(
      0,
      neighbourhoodSize
    );
    //Computing the predicted rating
    topNeighbors.forEach(([similarity, r_bp, r_bavg]) => {
      numerator += similarity * (r_bp - r_bavg);
      denominator += similarity;
    });
    let prediction = r_aavg + numerator / denominator;

    if (denominator === 0) {
      return r_aavg;
    }
    if (isNaN(prediction)) {
      return r_aavg;
    }
    if (prediction > 5) {
      this.rGreaterFive++;
      return 5;
    }
    if (prediction < 1) {
      this.rLessThanOne++;
      return 1;
    }
    return prediction;
  }

  //Updates the ratings matrix by computing the predicted ratings for all users and items
  updateMatrix() {
    //parse the number of users and items from the data
    const N = parseInt(this.data[0][0]);
    const M = parseInt(this.data[0][1]);
    //parse the ratings matrix from the data
    const ratings = this.data.slice(3).map((row) => row.map(Number));

    const MAE = { totalErrors: 0, predictionsMade: 0 };

    //Computing the average rating for each user
    const userAverages = this.preComputeAverages(ratings);
    //Initializing the results matrix
    let results = this.createMatrix(N, M);

    //Computing the predicted rating for each user and item
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < M; j++) {
        //If the rating is not 0, compute the predicted rating this is used to implement the leave-one-out cross validation
        if (ratings[i][j] !== 0) {
          //Storing the average rating of user i
          let tempAvg = userAverages[i].avg;
          //Updating the average rating of user i by removing the rating of item j
          let sum = userAverages[i].sum - ratings[i][j];
          let count = userAverages[i].count - 1;
          userAverages[i].avg = count === 0 ? 0 : sum / count;

          //Storing the rating of item j
          let tempRating = ratings[i][j];
          //Removing the rating of item j from the ratings matrix
          ratings[i][j] = 0;

          //Computing the similarity between user i and all other users
          let userSimilarties = this.computeSimilarities(
            ratings,
            userAverages,
            N,
            M,
            i
          );
          //Computing the predicted rating for user i and item j
          results[i][j] = parseFloat(
            this.prediction(
              i,
              j,
              ratings,
              this.neighborhoodSize,
              userSimilarties,
              userAverages
            ).toFixed(2)
          );

          //Restoring the rating of item j in the ratings matrix
          ratings[i][j] = tempRating;

          //Restoring the average rating of user i
          userAverages[i].avg = tempAvg;

          //Computing the mean absolute error between the predicted rating and the actual rating
          const diff = Math.abs(results[i][j] - ratings[i][j]);
          MAE.totalErrors += diff;
          MAE.predictionsMade++;
        } else {
          results[i][j] = ratings[i][j];
        }
      }
    }

    this.MAE = MAE.totalErrors / MAE.predictionsMade;
    this.predictionsMade = MAE.predictionsMade;
  }
}

class Itembased {
  // Constructor initializes the class with data and algorithm parameters
  constructor(data, neighborhoodSize, similarityThreshold, absoluteSimilarity) {
    this.data = data;
    this.neighborhoodSize = neighborhoodSize;
    this.similarityThreshold = similarityThreshold;
    this.totalNeighbours = 0;
    this.noValidNeighbors = 0;
    this.rGreaterFive = 0;
    this.rLessThanOne = 0;
    this.MAE = 0;
    this.predictionsMade = 0;
    this.absoluteSimilarity = absoluteSimilarity;
  }

  // Computes the average rating for each user
  preComputeAverages(ratings) {
    // Map through each user's ratings to calculate their average rating
    return ratings.map((userRatings) => {
      let sum = 0;
      let count = 0;
      userRatings.forEach((rating) => {
        if (rating !== 0) {
          sum += parseFloat(rating);
          count++;
        }
      });
      return { sum: sum, count: count, avg: count === 0 ? 0 : sum / count };
    });
  }

  // Creates a matrix of size n x m filled with zeros
  createMatrix(n, m) {
    return Array.from({ length: n }, () => Array(m).fill(0));
  }

  // Computes the similarity between items
  computeItemSimilaritiesForItem(ratings, userAverages, itemIndex) {
    const numItems = ratings[0].length;
    let itemSimilarities = Array(numItems).fill(0); //Intializing array to store similarities
    //iterate through all items
    for (let j = 0; j < numItems; j++) {
      if (j !== itemIndex) {
        //exclude the item itself
        let similarity = this.cosineSimilarity(
          itemIndex,
          j,
          ratings,
          userAverages
        );
        itemSimilarities[j] = similarity;
      }
    }

    return itemSimilarities;
  }

  //Computes the cosine similarity between two items
  cosineSimilarity(i, j, ratings, userAverages) {
    let numerator = 0;
    let denominatorI = 0;
    let denominatorJ = 0;
    //iterate through all users
    for (let u = 0; u < ratings.length; u++) {
      const userRatings = ratings[u];
      const r_ui = userRatings[i]; //rating of user u for item i
      const r_uj = userRatings[j]; //rating of user u for item j
      //if user u has rated both items i and j
      if (r_ui !== 0 && r_uj !== 0) {
        const r_uavg = userAverages[u].avg; //average rating of user u
        const diffI = r_ui - r_uavg; //difference between rating of user u for item i and average rating of user u
        const diffJ = r_uj - r_uavg; //difference between rating of user u for item j and average rating of user u
        numerator += diffI * diffJ;
        denominatorI += diffI * diffI;
        denominatorJ += diffJ * diffJ;
      }
    }

    if (denominatorI === 0 || denominatorJ === 0) return 0;
    return numerator / (Math.sqrt(denominatorI) * Math.sqrt(denominatorJ));
  }

  // Predicts the rating for a specific user and item
  predictRating(
    userIndex,
    itemIndex,
    ratings,
    neighbourhoodSize,
    similarities,
    userAverages
  ) {
    let r_uavg = userAverages[userIndex].avg;
    let numerator = 0;
    let denominator = 0;

    //Identifying the neighbours of item p based on the similarity
    let ratedItemsWithSimilarity = [];
    for (let i = 0; i < ratings[userIndex].length; i++) {
      let tempSim = similarities[i];
      if (this.absoluteSimilarity) {
        similarities[i] = Math.abs(similarities[i]);
      }
      if (
        ratings[userIndex][i] !== 0 &&
        similarities[i] > this.similarityThreshold &&
        i !== itemIndex
      ) {
        ratedItemsWithSimilarity.push([tempSim, ratings[userIndex][i]]);
      }
      similarities[i] = tempSim;
    }

    //If there are no valid neighbours, return the average rating of user a
    if (ratedItemsWithSimilarity.length === 0) {
      this.noValidNeighbors++;
      return r_uavg;
    }
    //If the number of neighbours is less than the specified neighbourhood size, use all the neighbours
    if (neighbourhoodSize > ratedItemsWithSimilarity.length) {
      neighbourhoodSize = ratedItemsWithSimilarity.length;
    }

    this.totalNeighbours += neighbourhoodSize;

    //Sorting the neighbours by similarity from highest to lowest
    ratedItemsWithSimilarity.sort((a, b) => b[0] - a[0]);
    let topNeighbors = ratedItemsWithSimilarity.slice(0, neighbourhoodSize);

    //Computing the predicted rating
    topNeighbors.forEach(([similarity, r_ui]) => {
      numerator += similarity * r_ui;
      denominator += Math.abs(similarity);
    });
    let prediction = numerator / denominator;

    if (denominator === 0) return r_uavg;
    if (isNaN(prediction)) {
      return r_uavg;
    }
    if (prediction > 5) {
      this.rGreaterFive++;
      return 5;
    }
    if (prediction < 1) {
      this.rLessThanOne++;
      return 1;
    }
    return prediction;
  }

  // Updates the ratings matrix by computing the predicted ratings for all users and items
  updateMatrix() {
    //parse the number of users and items from the data
    const N = parseInt(this.data[0][0]);
    const M = parseInt(this.data[0][1]);
    //parse the ratings matrix from the data
    const ratings = this.data.slice(3).map((row) => row.map(Number));
    const MAE = { totalErrors: 0, predictionsMade: 0 };

    //Computing the average rating for each user
    const userAverages = this.preComputeAverages(ratings);
    //Initializing the results matrix
    let results = this.createMatrix(N, M);

    // Iterate through all users and items to compute the predicted rating for each user and item
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < M; j++) {
        //If the rating is not 0, compute the predicted rating this is used to implement the leave-one-out cross validation
        if (ratings[i][j] !== 0) {
          //Storing the average rating of user i
          let tempAvg = userAverages[i].avg;

          //Updating the average rating of user i by removing the rating of item j
          let sum = userAverages[i].sum - ratings[i][j];
          let count = userAverages[i].count - 1;
          userAverages[i].avg = count === 0 ? 0 : sum / count;

          //Storing the rating of item j
          let tempRating = ratings[i][j];
          ratings[i][j] = 0;

          //Computing the similarity between item j and all other items
          let itemSimilarities = this.computeItemSimilaritiesForItem(
            ratings,
            userAverages,
            j
          );

          //Computing the predicted rating for user i and item j
          results[i][j] = parseFloat(
            this.predictRating(
              i,
              j,
              ratings,
              this.neighborhoodSize,
              itemSimilarities,
              userAverages
            ).toFixed(2)
          );

          //Restoring the rating of item j in the ratings matrix
          ratings[i][j] = tempRating;
          //Restoring the average rating of user i
          userAverages[i].avg = tempAvg;

          //Computing the mean absolute error between the predicted rating and the actual rating
          const diff = Math.abs(results[i][j] - ratings[i][j]);
          MAE.totalErrors += diff;
          MAE.predictionsMade++;
        } else {
          results[i][j] = ratings[i][j];
        }
      }
    }

    this.MAE = MAE.totalErrors / MAE.predictionsMade;
    this.predictionsMade = MAE.predictionsMade;
  }
}

// Loads the data from the specified file path
function loadFile(filePath) {
  try {
    const fileContent = fs.readFileSync(filePath, "utf8");
    const lines = fileContent.trim().split("\n");
    const data = lines.map((line) => line.trim().split(" "));
    return data;
  } catch (error) {
    console.error("Error reading or parsing the file:", error);
    return null;
  }
}

// Logs the results of the experiment to the console and to a log file
function logResults(experimentTitle, userBased, runTime) {
  const logData = `
  Experiment: ${experimentTitle}
  Total Predictions: ${userBased.predictionsMade}
  Predictions Less Than 1: ${userBased.rLessThanOne}
  Predictions Greater Than 5: ${userBased.rGreaterFive}
  No Valid Neighbours Cases: ${userBased.noValidNeighbors}
  Average Neighbourhood Size: ${
    userBased.totalNeighbours / userBased.predictionsMade
  }
  MAE: ${userBased.MAE}
  Run Time: ${runTime.toFixed(2)} seconds
  -----------------------------------
  `;

  if (experimentTitle.includes("user-based")) {
    fs.appendFileSync("userBasedLog.txt", logData, "utf8");
  } else {
    fs.appendFileSync("itemBasedLog.txt", logData, "utf8");
  }
}

// Runs the experiment for the user-based collaborative filtering recommendation system
function userBased() {
  const data = loadFile("assignment2-data.txt");

  for (
    let neighborhoodSize = 0;
    neighborhoodSize <= 100;
    neighborhoodSize += 5
  ) {
    if (neighborhoodSize === 0) continue;
    const start = Date.now();
    const userBased = new UserBased(data, neighborhoodSize, 0, false);
    userBased.updateMatrix();
    const end = Date.now();
    const runTime = (end - start) / 1000;

    logResults(
      `user-based, ${neighborhoodSize} neighbours, ignore negative similarities`,
      userBased,
      runTime
    );
  }

  for (let threshold = 0; threshold <= 1; threshold += 0.1) {
    const start = Date.now();
    const userBased = new UserBased(data, 25, threshold, false);
    userBased.updateMatrix();
    const end = Date.now();
    const runTime = (end - start) / 1000;

    logResults(
      `user-based, 25 neighbours, ignore similarities less then ${threshold.toFixed(
        1
      )}`,
      userBased,
      runTime
    );
  }

  for (let threshold = 0; threshold <= 1; threshold += 0.1) {
    const start = Date.now();
    const userBased = new UserBased(data, 25, threshold, true);
    userBased.updateMatrix();
    const end = Date.now();
    const runTime = (end - start) / 1000;

    logResults(
      `user-based, 25 neighbours, take absolute value of similarities, ignore similarities less then ${threshold.toFixed(
        1
      )}`,
      userBased,
      runTime
    );
  }

  for (
    let neighbourhoodSize = 0;
    neighbourhoodSize <= 100;
    neighbourhoodSize += 5
  ) {
    if (neighbourhoodSize === 0) continue;
    const start = Date.now();
    const userBased = new UserBased(data, neighbourhoodSize, 0, true);
    userBased.updateMatrix();
    const end = Date.now();
    const runTime = (end - start) / 1000;

    logResults(
      `user-based, ${neighbourhoodSize} neighbours, take absolute similarities,ignore negative similarities`,
      userBased,
      runTime
    );
  }
}

// Runs the experiment for the item-based collaborative filtering recommendation system
function itemBased() {
  const data = loadFile("assignment2-data.txt");

  for (
    let neighborhoodSize = 0;
    neighborhoodSize <= 500;
    neighborhoodSize += 25
  ) {
    if (neighborhoodSize === 0) continue;
    const start = Date.now();
    const itemBased = new Itembased(data, neighborhoodSize, 0, false);
    itemBased.updateMatrix();
    const end = Date.now();
    const runTime = (end - start) / 1000;

    logResults(
      `item-based, ${neighborhoodSize} neighbours, ignore negative similarities`,
      itemBased,
      runTime
    );
  }

  for (let threshold = 0; threshold <= 1; threshold += 0.1) {
    const start = Date.now();
    const itemBased = new Itembased(data, 300, threshold, false);
    itemBased.updateMatrix();
    const end = Date.now();
    const runTime = (end - start) / 1000;

    logResults(
      `item-based, 300 neighbours, ignore similarities less then ${threshold.toFixed(
        1
      )}`,
      itemBased,
      runTime
    );
  }

  for (let threshold = 0; threshold <= 1; threshold += 0.1) {
    const start = Date.now();
    const itemBased = new Itembased(data, 300, threshold, true);
    itemBased.updateMatrix();
    const end = Date.now();
    const runTime = (end - start) / 1000;

    logResults(
      `item-based, 300 neighbours, take absolute value of similarities, ignore similarities less then ${threshold.toFixed(
        1
      )}`,
      itemBased,
      runTime
    );
  }

  for (
    let neighborhoodSize = 0;
    neighborhoodSize <= 500;
    neighborhoodSize += 25
  ) {
    if (neighborhoodSize === 0) continue;
    const start = Date.now();
    const itemBased = new Itembased(data, neighborhoodSize, 0, true);
    itemBased.updateMatrix();
    const end = Date.now();
    const runTime = (end - start) / 1000;

    logResults(
      `item-based, ${neighborhoodSize} neighbours, take absolute similarities,ignore negative similarities`,
      itemBased,
      runTime
    );
  }
}

// Run the experiments
itemBased();
userBased();
