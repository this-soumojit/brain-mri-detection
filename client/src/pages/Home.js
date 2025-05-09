
// import React, { useState } from "react";
// import axios from "axios";
// import { useNavigate } from "react-router-dom";
// import {
//   Button,
//   Container,
//   Typography,
//   Box,
//   LinearProgress,
//   Card,
//   CardMedia,
//   IconButton,
//   Alert,
// } from "@mui/material";
// import UploadIcon from "@mui/icons-material/Upload";
// import CloseIcon from "@mui/icons-material/Close";
// import ReplayIcon from "@mui/icons-material/Replay";

// const Home = () => {
//   const [file, setFile] = useState(null);
//   const [previewUrl, setPreviewUrl] = useState("");
//   const [isUploading, setIsUploading] = useState(false);
//   const [uploadProgress, setUploadProgress] = useState(0);
//   const [error, setError] = useState(null);
//   const navigate = useNavigate();

//   const handleFileChange = (e) => {
//     const selectedFile = e.target.files[0];
//     if (!selectedFile) return;

//     // Validate file type
//     const validTypes = [
//       "image/dicom",
//       "image/nifti",
//       "image/jpeg",
//       "image/png",
//       "application/dicom",
//     ];
//     if (
//       !validTypes.includes(selectedFile.type) &&
//       !selectedFile.name.match(/\.(dcm|nii|jpg|jpeg|png)$/i)
//     ) {
//       setError(
//         "Unsupported file format. Please upload DICOM, NIfTI, JPEG, or PNG."
//       );
//       return;
//     }

//     setFile(selectedFile);
//     setError(null);

//     // Create preview for image files
//     if (
//       selectedFile.type.match("image.*") &&
//       !selectedFile.type.includes("dicom")
//     ) {
//       const reader = new FileReader();
//       reader.onload = (event) => setPreviewUrl(event.target.result);
//       reader.readAsDataURL(selectedFile);
//     } else {
//       setPreviewUrl("");
//     }
//   };

//   const handleClearFile = () => {
//     setFile(null);
//     setPreviewUrl("");
//     setError(null);
//   };

//   const handleUpload = async () => {
//     if (!file) {
//       setError("Please select a file first");
//       return;
//     }

//     const formData = new FormData();
//     formData.append("mriScan", file); // Must match backend field name

//     try {
//       setIsUploading(true);
//       setError(null);
//       setUploadProgress(0);

//       const response = await axios.post(
//         "http://localhost:5000/api/upload",
//         formData,
//         {
//           onUploadProgress: (progressEvent) => {
//             const progress = Math.round(
//               (progressEvent.loaded * 100) / progressEvent.total
//             );
//             setUploadProgress(progress);
//           },
//           headers: {
//             "Content-Type": "multipart/form-data",
//           },
//           timeout: 60000, // 30 seconds timeout
//         }
//       );

//       console.log("API Response:", response.data); // Debug log

//       if (!response.data.success) {
//         throw new Error(response.data.message || "Prediction failed");
//       }

//       // Navigate with all available data
//       navigate("/results", {
//         state: {
//           prediction:
//             response.data.prediction ||
//             response.data.data?.prediction ||
//             "No prediction available",
//           confidence:
//             response.data.confidence || response.data.data?.confidence,
//           image: previewUrl,
//           rawData: response.data, // For debugging
//         },
//         replace: true,
//       });
//     } catch (err) {
//       console.error("Upload error:", err.response?.data || err.message);
//       setError(
//         err.response?.data?.message ||
//           err.message ||
//           "Analysis failed. Please try again."
//       );
//     } finally {
//       setIsUploading(false);
//     }
//   };

//   return (
//     <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
//       <Typography
//         variant="h3"
//         component="h1"
//         gutterBottom
//         align="center"
//         sx={{ fontWeight: 700 }}
//       >
//         Brain MRI Analysis
//       </Typography>

//       <Typography
//         variant="subtitle1"
//         align="center"
//         color="text.secondary"
//         sx={{ mb: 4 }}
//       >
//         Upload MRI scans for tumor detection and classification
//       </Typography>

//       <Card sx={{ p: 3, mb: 3, border: "1px dashed #1976d2" }}>
//         <input
//           accept=".dcm,.nii,.jpg,.jpeg,.png"
//           style={{ display: "none" }}
//           id="mri-upload-input"
//           type="file"
//           onChange={handleFileChange}
//           disabled={isUploading}
//         />
//         <label htmlFor="mri-upload-input">
//           <Button
//             variant="contained"
//             component="span"
//             startIcon={<UploadIcon />}
//             size="large"
//             disabled={isUploading}
//             fullWidth
//             sx={{ py: 2 }}
//           >
//             Select MRI File
//           </Button>
//         </label>

//         {file && (
//           <Box sx={{ mt: 2, display: "flex", alignItems: "center" }}>
//             <Typography variant="body1" sx={{ flexGrow: 1 }}>
//               {file.name} ({Math.round(file.size / 1024)} KB)
//             </Typography>
//             <IconButton
//               onClick={handleClearFile}
//               disabled={isUploading}
//               color="error"
//             >
//               <CloseIcon />
//             </IconButton>
//           </Box>
//         )}

//         {previewUrl && (
//           <Box sx={{ mt: 3, textAlign: "center" }}>
//             <CardMedia
//               component="img"
//               image={previewUrl}
//               alt="Preview"
//               sx={{
//                 maxHeight: 300,
//                 maxWidth: "100%",
//                 objectFit: "contain",
//                 borderRadius: 1,
//               }}
//             />
//           </Box>
//         )}

//         <Button
//           variant="contained"
//           color="primary"
//           onClick={handleUpload}
//           disabled={!file || isUploading}
//           fullWidth
//           sx={{ mt: 3, py: 2 }}
//         >
//           {isUploading ? (
//             <>
//               Analyzing... {uploadProgress}%
//               <LinearProgress
//                 variant="determinate"
//                 value={uploadProgress}
//                 sx={{ ml: 2, width: "100px" }}
//               />
//             </>
//           ) : (
//             "Analyze Scan"
//           )}
//         </Button>

//         {error && (
//           <Alert severity="error" sx={{ mt: 2 }}>
//             {error}
//           </Alert>
//         )}
//       </Card>

//       <Card sx={{ p: 3 }}>
//         <Typography variant="h6" gutterBottom>
//           Instructions
//         </Typography>
//         <ul style={{ paddingLeft: "20px" }}>
//           <li>Supported formats: DICOM (.dcm), NIfTI (.nii), JPEG/PNG</li>
//           <li>Minimum resolution: 128×128 pixels</li>
//           <li>Maximum file size: 20MB</li>
//           <li>Analysis typically takes 10-30 seconds</li>
//         </ul>
//       </Card>

//       <Button
//         variant="contained"
//         size="large"
//         onClick={() => navigate("/evaluation-report")}
//         sx={{
//           mt: 4,
//           background: "#00bcd4",
//           color: "white",
//           padding: "10px 24px",
//           fontSize: "1rem",
//           display: "block",
//           mx: "auto",
//           "&:hover": {
//             background: "#0097a7",
//           },
//         }}
//       >
//         Get Evaluation Report
//       </Button>
//     </Container>
//   );
  
// };

// export default Home;


























import React, { useState } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import {
  Button,
  // Container,
  Typography,
  Box,
  CircularProgress,
  Card,
  CardMedia,
  IconButton,
  Alert,
} from "@mui/material";
import UploadIcon from "@mui/icons-material/Upload";
import CloseIcon from "@mui/icons-material/Close";
import brainImage from "../assets/bg.jpg"; // Use the same background as the landing page

const colors = {
  primary: "#00bcd4", // Slightly muted blue/teal to complement the background
  secondary: "#ff4081", // Accent color for hover effects
  text: "#f5f5f5", // Light text color for better contrast
};

const Home = () => {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;

    // Validate file type
    const validTypes = [
      "image/dicom",
      "image/nifti",
      "image/jpeg",
      "image/png",
      "application/dicom",
    ];
    if (
      !validTypes.includes(selectedFile.type) &&
      !selectedFile.name.match(/\.(dcm|nii|jpg|jpeg|png)$/i)
    ) {
      setError(
        "Unsupported file format. Please upload DICOM, NIfTI, JPEG, or PNG."
      );
      return;
    }

    setFile(selectedFile);
    setError(null);

    // Create preview for image files
    if (
      selectedFile.type.match("image.*") &&
      !selectedFile.type.includes("dicom")
    ) {
      const reader = new FileReader();
      reader.onload = (event) => setPreviewUrl(event.target.result);
      reader.readAsDataURL(selectedFile);
    } else {
      setPreviewUrl("");
    }
  };

  const handleClearFile = () => {
    setFile(null);
    setPreviewUrl("");
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a file first");
      return;
    }

    const formData = new FormData();
    formData.append("mriScan", file); // Must match backend field name

    try {
      setIsUploading(true);
      setError(null);
      setUploadProgress(0);

      const response = await axios.post(
        "http://localhost:5000/api/upload",
        formData,
        {
          onUploadProgress: (progressEvent) => {
            const progress = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(progress);
          },
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 60000, // 30 seconds timeout
        }
      );

      console.log("API Response:", response.data); // Debug log

      if (!response.data.success) {
        throw new Error(response.data.message || "Prediction failed");
      }

      // Navigate with all available data
      navigate("/results", {
        state: {
          prediction:
            response.data.prediction ||
            response.data.data?.prediction ||
            "No prediction available",
          confidence:
            response.data.confidence || response.data.data?.confidence,
          image: previewUrl,
          rawData: response.data, // For debugging
        },
        replace: true,
      });
    } catch (err) {
      console.error("Upload error:", err.response?.data || err.message);
      setError(
        err.response?.data?.message ||
          err.message ||
          "Analysis failed. Please try again."
      );
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh", // Ensure the container takes the full screen height
        background: `linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.7)), url(${brainImage})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
      }}
    >
      <Box
        sx={{
          background: "rgba(0, 0, 0, 0.6)",
          borderRadius: 2,
          p: 4,
          width: "100%",
          maxWidth: 600,
          boxShadow: "0px 4px 20px rgba(0, 0, 0, 0.3)",
        }}
      >
        <Typography
          variant="h3"
          component="h1"
          gutterBottom
          align="center"
          sx={{
            fontWeight: 700,
            color: colors.text,
            mb: 3,
            textAlign: "center",
            background: `linear-gradient(45deg, ${colors.primary}, ${colors.secondary})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          Brain MRI Analysis
        </Typography>

        <Typography
          variant="subtitle1"
          align="center"
          color="text.secondary"
          sx={{ mb: 4 }}
        >
          Upload MRI scans for tumor detection and classification
        </Typography>

        <Card sx={{ p: 3, mb: 3, borderRadius: 2, backgroundColor: "#333" }}>
          <input
            accept=".dcm,.nii,.jpg,.jpeg,.png"
            style={{ display: "none" }}
            id="mri-upload-input"
            type="file"
            onChange={handleFileChange}
            disabled={isUploading}
          />
          <label htmlFor="mri-upload-input">
            <Button
              variant="contained"
              component="span"
              startIcon={<UploadIcon />}
              size="large"
              disabled={isUploading}
              fullWidth
              sx={{
                py: 2,
                background: "#0097a7", // Slightly muted teal to blend better
                "&:hover": {
                  background: "#007c8a", // Darker shade on hover
                },
                borderRadius: 1.5, // Rounded corners
                boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.2)", // Subtle shadow
              }}
            >
              Select MRI File
            </Button>
          </label>

          {file && (
            <Box sx={{ mt: 2, display: "flex", alignItems: "center" }}>
              <Typography variant="body1" sx={{ flexGrow: 1 }}>
                {file.name} ({Math.round(file.size / 1024)} KB)
              </Typography>
              <IconButton
                onClick={handleClearFile}
                disabled={isUploading}
                color="error"
              >
                <CloseIcon />
              </IconButton>
            </Box>
          )}

          {previewUrl && (
            <Box
              sx={{
                mt: 3,
                textAlign: "center",
                backgroundColor: "transparent", // Remove background overlay
                padding: "10px",
              }}
            >
              <CardMedia
                component="img"
                image={previewUrl}
                alt="Preview"
                sx={{
                  maxHeight: 300,
                  maxWidth: "100%",
                  objectFit: "contain",
                  borderRadius: 1,
                }}
              />
            </Box>
          )}

          <Button
            variant="contained"
            color="primary"
            onClick={handleUpload}
            disabled={!file || isUploading}
            fullWidth
            sx={{
              mt: 3,
              py: 2,
              background: "#0097a7", // Match the theme color
              "&:hover": {
                background: "#007c8a", // Darker shade on hover
              },
              borderRadius: 1.5,
              boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.2)", // Subtle shadow
            }}
          >
            {isUploading ? (
              <>
                Analyzing... {uploadProgress}%
                <CircularProgress
                  size={24}
                  sx={{ color: "white", ml: 2 }}
                  thickness={5}
                />
              </>
            ) : (
              "Analyze Scan"
            )}
          </Button>

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}
        </Card>

        <Card
          sx={{
            p: 3,
            background: "rgba(0, 0, 0, 0.6)",
            borderRadius: 2,
            mt: 4,
            boxShadow: "0px 4px 15px rgba(0, 0, 0, 0.4)",
          }}
        >
          <Typography
            variant="h6"
            gutterBottom
            sx={{ color: colors.text, textAlign: "center" }}
          >
            Instructions
          </Typography>
          <ul style={{ paddingLeft: "20px", color: colors.text }}>
            <li>Supported formats: DICOM (.dcm), NIfTI (.nii), JPEG/PNG</li>
            <li>Minimum resolution: 128×128 pixels</li>
            <li>Maximum file size: 20MB</li>
            <li>Analysis typically takes 10-30 seconds</li>
          </ul>
        </Card>

        <Button
          variant="contained"
          size="large"
          onClick={() => navigate("/evaluation-report")}
          sx={{
            mt: 4,
            background: colors.primary,
            color: "white",
            padding: "10px 24px",
            fontSize: "1rem",
            display: "block",
            mx: "auto",
            "&:hover": {
              background: "#0097a7",
            },
          }}
        >
          Get Evaluation Report
        </Button>
        
      </Box>
    </Box>
  );
};

export default Home;
