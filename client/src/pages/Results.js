import React from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  Container,
  Typography,
  Button,
  Box,
  Card,
  CardContent,
  CardMedia,
  Chip,
  CircularProgress,
  Alert,
  Divider,
  Stack,
} from "@mui/material";
import ReplayIcon from "@mui/icons-material/Replay";
import MedicalInformationIcon from "@mui/icons-material/MedicalInformation";
import ImageIcon from "@mui/icons-material/Image";

const Results = () => {
  const { state } = useLocation();
  const navigate = useNavigate();

  // Debugging logs
  console.log("Location state:", state);
  console.log("Raw response data:", state?.rawData);

  if (!state?.prediction) {
    return (
      <Container maxWidth="md" sx={{ mt: 4, textAlign: "center" }}>
        <Alert severity="error" sx={{ mb: 3 }}>
          <Typography variant="h6" component="div">
            No Results Found
          </Typography>
          <Typography variant="body1">
            The prediction results were not properly received.
          </Typography>
        </Alert>

        {state?.rawData && (
          <Card sx={{ mt: 3, p: 2, textAlign: "left" }}>
            <Typography variant="subtitle2" gutterBottom>
              Debug Information:
            </Typography>
            <Box
              component="pre"
              sx={{
                overflowX: "auto",
                p: 1,
                bgcolor: "#f5f5f5",
                borderRadius: 1,
              }}
            >
              {JSON.stringify(state.rawData, null, 2)}
            </Box>
          </Card>
        )}

        <Button
          variant="contained"
          startIcon={<ReplayIcon />}
          onClick={() => navigate("/")}
          sx={{ mt: 3 }}
        >
          Try Again
        </Button>
      </Container>
    );
  }

  // Format confidence percentage
  const confidencePercentage = state.confidence
    ? `${(state.confidence * 100).toFixed(1)}%`
    : "Not available";

  // Get appropriate color based on prediction
  const getPredictionColor = () => {
    switch (state.prediction.toLowerCase()) {
      case "no_tumor":
        return "success";
      case "glioma":
      case "meningioma":
      case "pituitary":
        return "warning";
      default:
        return "primary";
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Card elevation={3}>
        <CardContent>
          <Typography
            variant="h4"
            component="h1"
            gutterBottom
            align="center"
            sx={{ fontWeight: 700 }}
          >
            Analysis Results
          </Typography>

          <Divider sx={{ my: 3 }} />

          {state.image && (
            <>
              <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                <ImageIcon color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">MRI Scan Preview</Typography>
              </Box>
              <CardMedia
                component="img"
                image={state.image}
                alt="MRI Scan"
                sx={{
                  maxHeight: 300,
                  width: "100%",
                  objectFit: "contain",
                  borderRadius: 1,
                  mb: 3,
                  border: "1px solid #ddd",
                }}
              />
            </>
          )}

          <Box
            sx={{
              p: 3,
              bgcolor: "#f9f9f9",
              borderRadius: 1,
              mb: 3,
            }}
          >
            <Stack
              direction="row"
              alignItems="center"
              spacing={1}
              sx={{ mb: 2 }}
            >
              <MedicalInformationIcon color="primary" />
              <Typography variant="h5">Diagnosis</Typography>
            </Stack>

            <Box
              sx={{
                display: "flex",
                flexDirection: { xs: "column", sm: "row" },
                alignItems: "center",
                gap: 3,
              }}
            >
              <Chip
                label={state.prediction.replace(/_/g, " ")}
                color={getPredictionColor()}
                sx={{
                  fontSize: "1.2rem",
                  px: 3,
                  py: 1,
                  textTransform: "capitalize",
                }}
              />

              <Box>
                <Typography variant="subtitle1" color="text.secondary">
                  Confidence Level
                </Typography>
                <Box sx={{ display: "flex", alignItems: "center", mt: 1 }}>
                  <CircularProgress
                    variant="determinate"
                    value={state.confidence ? state.confidence * 100 : 0}
                    size={24}
                    thickness={6}
                    sx={{ mr: 2 }}
                  />
                  <Typography variant="h6">{confidencePercentage}</Typography>
                </Box>
              </Box>
            </Box>
          </Box>

          <Box sx={{ mt: 4, textAlign: "center" }}>
            <Button
              variant="contained"
              size="large"
              startIcon={<ReplayIcon />}
              onClick={() => navigate("/home")}
              sx={{ px: 4, py: 1.5 }}
            >
              Analyze Another Scan
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Debug section (visible in development) */}
      {process.env.NODE_ENV === "development" && state?.rawData && (
        <Box sx={{ mt: 4 }}>
          <Typography variant="h6" gutterBottom>
            Debug Information
          </Typography>
          <Box
            component="pre"
            sx={{
              p: 2,
              bgcolor: "#f5f5f5",
              borderRadius: 1,
              overflowX: "auto",
            }}
          >
            {JSON.stringify(state.rawData, null, 2)}
          </Box>
        </Box>
      )}
    </Container>
  );
};

export default Results;





