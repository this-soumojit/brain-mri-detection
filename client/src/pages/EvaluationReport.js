import React from "react";
import { Box, Typography, Container } from "@mui/material";

const EvaluationReport = () => {
  return (
    <Box
      sx={{
        backgroundColor: "#121212",
        color: "#f5f5f5",
        minHeight: "100vh",
        padding: "20px",
      }}
    >
      <Container maxWidth="md">
        <Typography
          variant="h4"
          sx={{
            fontWeight: 700,
            mb: 2,
            background: `linear-gradient(45deg, #00bcd4, #ff4081)`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            display: "inline-block",
          }}
        >
          Model Evaluation Report
        </Typography>

        <Box
          sx={{
            background: "#1e1e1e",
            padding: "20px",
            borderRadius: "8px",
            marginTop: "20px",
          }}
        >
          <Typography
            variant="h6"
            sx={{ fontWeight: 600, marginBottom: "10px" }}
          >
            [RESULTS]
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            Test Loss: 0.6687
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            Test Accuracy: 93.44%
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            Test AUC: 98.78%
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            Precision: 93.57%
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            Recall: 93.29%
          </Typography>
        </Box>

        <Box
          sx={{
            background: "#1e1e1e",
            padding: "20px",
            borderRadius: "8px",
            marginTop: "20px",
          }}
        >
          <Typography
            variant="h6"
            sx={{ fontWeight: 600, marginBottom: "10px" }}
          >
            [CLASSIFICATION REPORT]
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            Glioma: Precision: 1.00, Recall: 0.75, F1-score: 0.85, Support: 300
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            Meningioma: Precision: 0.95, Recall: 0.97, F1-score: 0.96, Support:
            306
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            Notumor: Precision: 1.00, Recall: 1.00, F1-score: 1.00, Support: 405
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            Pituitary: Precision: 0.81, Recall: 0.99, F1-score: 0.89, Support:
            300
          </Typography>
        </Box>

        <Box
          sx={{
            background: "#1e1e1e",
            padding: "20px",
            borderRadius: "8px",
            marginTop: "20px",
          }}
        >
          <Typography
            variant="h6"
            sx={{ fontWeight: 600, marginBottom: "10px" }}
          >
            [CONFUSION MATRIX]
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            [[224 15 0 61]
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            [ 0 298 0 8]
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            [ 0 0 405 0]
          </Typography>
          <Typography variant="body1" sx={{ marginBottom: "10px" }}>
            [ 0 2 0 298]]
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default EvaluationReport;
