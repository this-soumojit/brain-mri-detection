// const mongoose = require("mongoose");

// const connectDB = async () => {
//   try {
//     await mongoose.connect("mongodb://localhost:27017/mri-scans", {
//       useNewUrlParser: true,
//       useUnifiedTopology: true,
//     });
//     console.log("MongoDB Connected");
//   } catch (err) {
//     console.error("MongoDB Connection Error:", err);
//     process.exit(1);
//   }
// };

// module.exports = connectDB;


const mongoose = require("mongoose");

const connectDB = async () => {
  try {
    // Replace with your Atlas connection string from Step 1
    const atlasUri =
      "mongodb+srv://soumojitghosh325:ghosh@cluster0.qljd2ov.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0";

    await mongoose.connect(atlasUri, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log("MongoDB Atlas Connected!");
  } catch (err) {  
    console.error("MongoDB Atlas Connection Error:", err);
    process.exit(1);
  }
};

module.exports = connectDB;