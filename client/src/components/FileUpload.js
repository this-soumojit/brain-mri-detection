// import React, { useState } from "react";
// import axios from "axios";
// import { useNavigate } from "react-router-dom";

// export default function FileUpload() {
//   const [file, setFile] = useState(null);
//   const navigate = useNavigate();

//   const handleUpload = async () => {
//     const formData = new FormData();
//     formData.append("mriScan", file);

//     try {
//       const res = await axios.post(
//         "http://localhost:5000/api/upload",
//         formData
//       );
//       navigate(`/results/${res.data.scanId}`);
//     } catch (err) {
//       alert("Upload failed!");
//     }
//   };

//   return (
//     <div>
//       <input type="file" onChange={(e) => setFile(e.target.files[0])} />
//       <button onClick={handleUpload}>Upload</button>
//     </div>
//   );
// }



















// import React, { useState } from "react";
// import axios from "axios";

// export default function FileUpload() {
//   const [file, setFile] = useState(null);
//   const [prediction, setPrediction] = useState(null); // To store result
//   const [loading, setLoading] = useState(false); // To manage loading state

//   const handleUpload = async () => {
//     const formData = new FormData();
//     formData.append("mriScan", file);

//     setLoading(true); // Start loading state

//     try {
//       const res = await axios.post(
//         "http://localhost:5000/api/upload",
//         formData
//       );
//       console.log("Backend Response:", res.data); // DEBUG
//       setPrediction(res.data.data.prediction); // store result to display
//     } catch (err) {
//       console.error("Upload failed:", err);
//       alert("Upload failed!");
//     } finally {
//       setLoading(false); // Stop loading state
//     }
//   };

//   return (
//     <div style={{ padding: "20px" }}>
//       <h2>Upload Brain MRI Image</h2>
//       <input type="file" onChange={(e) => setFile(e.target.files[0])} />
//       <button onClick={handleUpload} disabled={!file || loading}>
//         {loading ? "Uploading..." : "Upload"}
//       </button>

//       {prediction && (
//         <div style={{ marginTop: "20px" }}>
//           <h3>Prediction Result:</h3>
//           <p style={{ fontSize: "20px", fontWeight: "bold" }}>{prediction}</p>
//         </div>
//       )}
//     </div>
//   );
// }





// import React, { useState } from "react";
// import axios from "axios";
// import { useNavigate } from "react-router-dom"; // Make sure you import useNavigate

// export default function FileUpload() {
//   const [file, setFile] = useState(null);
//   const [loading, setLoading] = useState(false); // To manage loading state
//   const navigate = useNavigate(); // Initialize navigate

//   const handleUpload = async () => {
//     const formData = new FormData();
//     formData.append("mriScan", file);

//     setLoading(true); // Start loading state

//     try {
//       const res = await axios.post(
//         "http://localhost:5000/api/upload",
//         formData
//       );
//       console.log("Backend Response:", res.data); // DEBUG
//       const predictionData = res.data.data.prediction; // Save the prediction data
//       console.log("Prediction Data:", res.data.data.prediction);


//       // Make sure predictionData is received before navigating
//       if (predictionData) {
//         // Navigate to the results page with the prediction data
//         navigate("/results", { state: { prediction: predictionData } });
//       } else {
//         alert("No prediction data received.");
//       }
//     } catch (err) {
//       console.error("Upload failed:", err);
//       alert("Upload failed!");
//     } finally {
//       setLoading(false); // Stop loading state
//     }
//   };

//   return (
//     <div style={{ padding: "20px" }}>
//       <h2>Upload Brain MRI Image</h2>
//       <input type="file" onChange={(e) => setFile(e.target.files[0])} />
//       <button onClick={handleUpload} disabled={!file || loading}>
//         {loading ? "Uploading..." : "Upload"}
//       </button>
//     </div>
//   );
// }










// import React, { useState } from "react";

// const FileUpload = () => {
//   const [file, setFile] = useState(null);
//   const [prediction, setPrediction] = useState("");
//   const [loading, setLoading] = useState(false);

//   const handleFileChange = (e) => {
//     setFile(e.target.files[0]);
//     setPrediction("");
//   };

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     if (!file) return;

//     const formData = new FormData();
//     formData.append("image", file);

//     setLoading(true);
//     setPrediction("");

//     try {
//       const response = await fetch("http://localhost:5000/upload", {
//         method: "POST",
//         body: formData,
//       });

//       const data = await response.json();
//       setPrediction(data.prediction || "No prediction received.");
//     } catch (error) {
//       console.error("Error during prediction:", error);
//       setPrediction("Error during prediction. Check the server.");
//     }

//     setLoading(false);
//   };

//   return (
//     <div style={{ textAlign: "center", padding: "40px" }}>
//       <h1>🧠 Brain MRI Disease Detector</h1>
//       <form onSubmit={handleSubmit}>
//         <input
//           type="file"
//           accept="image/*"
//           onChange={handleFileChange}
//           required
//         />
//         <br />
//         <button type="submit" style={{ marginTop: "20px" }}>
//           Predict
//         </button>
//       </form>

//       <div style={{ marginTop: "30px", fontSize: "18px" }}>
//         {loading && <span>⏳ Processing...</span>}
//         {prediction && !loading && <span>✅ Prediction: {prediction}</span>}
//       </div>
//     </div>
//   );
// };

// export default FileUpload;
