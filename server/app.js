// const express = require("express");
// const cors = require("cors");
// const multer = require("multer");
// const { PythonShell } = require("python-shell");
// const connectDB = require("./config/db"); // Your db.js connection
// const Scan = require("./models/Scan"); // Your improved Scan model

// const app = express();
// const upload = multer({ dest: "uploads/" });

// // Middleware
// app.use(cors());
// app.use(express.json());

// // Initialize Database Connection
// connectDB(); // Uses your Atlas connection from db.js

// // Enhanced Upload Route
// app.post("/api/upload", upload.single("mriScan"), async (req, res) => {
//   try {
//     const newScan = new Scan({
//       filename: req.file.originalname,
//       fileUrl: `/uploads/${req.file.filename}`,
//       storagePath: "local", // Change to "s3" when using cloud storage
//       scanType: determineScanType(req.file), // Implement this function
//       patientId: extractPatientId(req.file), // Implement this function
//     });

//     await newScan.save();

//     res.status(201).json({
//       success: true,
//       message: "MRI scan uploaded successfully",
//       data: {
//         scanId: newScan._id,
//         filename: newScan.filename,
//         scanType: newScan.scanType,
//       },
//     });
//   } catch (err) {
//     res.status(500).json({
//       success: false,
//       error: err.message,
//       details: process.env.NODE_ENV === "development" ? err.stack : undefined,
//     });
//   }
// });

// // Helper Functions (implement these)
// function determineScanType(file) {
//   // Analyze filename or content to determine scan type (T1, T2, etc.)
//   return "T1"; // Default value - implement your logic
// }

// function extractPatientId(file) {
//   // Extract patient ID from filename or metadata
//   return "PAT-" + Math.random().toString(36).substring(2, 9).toUpperCase(); // Temp random ID
// }

// // Start Server
// const PORT = process.env.PORT || 5000;
// app.listen(PORT, () => {
//   console.log(` Server running on http://localhost:${PORT}`);
//   console.log(`Temporary uploads directory: ${__dirname}/uploads`);
// });








const express = require("express");
const cors = require("cors");
const multer = require("multer");
const path = require("path");
const { exec } = require("child_process");
const sharp = require("sharp");
const fs = require("fs");

const app = express();
const upload = multer({ dest: "uploads/" });

// Middleware
app.use(cors());
app.use(express.json());

// Upload Route
app.post("/api/upload", upload.single("mriScan"), async (req, res) => {
  try {
    const uploadedImagePath = path.join(
      __dirname,
      "uploads",
      req.file.filename
    );
    const preprocessedImagePath = path.join(
      __dirname,
      "uploads",
      `preprocessed_${req.file.filename}.jpg`
    );

    console.log("📤 Uploaded image path:", uploadedImagePath);

    // Preprocess with sharp
    await sharp(uploadedImagePath)
      .resize(124,124)
      .toFormat("jpeg")
      .toFile(preprocessedImagePath);

    console.log("🛠️ Preprocessed image saved to:", preprocessedImagePath);

    // Running prediction with Python script
    console.log("Running prediction with Python script...");

    // Use exec to run the Python script directly with double quotes around the paths
    exec(
      `python "${path.join(
        __dirname,
        "..",
        "model",
        "predict.py"
      )}" "${preprocessedImagePath}"`,
      { timeout: 60000 },
      (err, stdout, stderr) => {
        if (err) {
          console.error("❌ Error running Python prediction script:", err);
          console.error("Python error details:", stderr);
          return res.status(500).json({ error: "Prediction failed." });
        }

        // Clean up Python script output (remove extra logs)
        const cleanedOutput = stdout
          .split("\n")
          .filter((line) => line.trim() !== "")
          .pop();
        console.log("✅ Python script output:", cleanedOutput);

        // Proceed with the rest of the code, assuming cleanedOutput contains the prediction in JSON format
        try {
          const predictionResult = JSON.parse(cleanedOutput);
          console.log("Parsed prediction:", predictionResult);

          return res.status(200).json({
            success: true,
            message: "Prediction successful",
            data: predictionResult,
          });
        } catch (parseError) {
          console.error("Error parsing prediction output:", parseError);
          return res
            .status(500)
            .json({ error: "Invalid prediction output format." });
        }
      }
    );
  } catch (err) {
    console.error("🚨 Upload error:", err);
    return res
      .status(500)
      .json({ success: false, error: "Upload failed due to server error." });
  }
});

// Start Server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`🚀 Server running at: http://localhost:${PORT}`);
  console.log(`📂 Uploads folder: ${path.join(__dirname, "uploads")}`);
});
