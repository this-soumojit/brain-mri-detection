import React from "react";
import {
  Box,
  Container,
  Typography,
  Grid,
  Button,
  IconButton,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import brainImage from "../assets/bg.jpg";
import logo from "../assets/brain.png";
import HomeIcon from "@mui/icons-material/Home"; // Add this import

const colors = {
  primary: "#00bcd4",
  secondary: "#ff4081",
  text: "#f5f5f5",
};

export default function HowItWorks() {
  const navigate = useNavigate();

  return (
    <Box
      sx={{
        minHeight: "100vh",
        background: `
          linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.7)),
          url(${brainImage})
        `,
        backgroundSize: "cover",
        backgroundPosition: "center",
        color: colors.text,
      }}
    >
      {/* Navbar */}
      <Box
        sx={{
          py: 1.5,
          backdropFilter: "blur(10px)",
          background: "rgba(0, 0, 0, 0.4)",
          borderBottom: `1px solid ${colors.primary}30`,
          width: "100vw",
          marginLeft: "calc(-50vw + 50%)",
        }}
      >
        <Container
          maxWidth="lg"
          sx={{ display: "flex", justifyContent: "space-between" }}
        >
          <img src={logo} alt="Logo" style={{ height: 50 }} />
          <Box sx={{ display: "flex", gap: 4 }}>
            <Button
              color="inherit"
              sx={{ color: colors.text }}
              onClick={() => navigate("/features")}
            >
              Features
            </Button>
            <Button
              color="inherit"
              sx={{ color: colors.text }}
              onClick={() => navigate("/about")}
            >
              About Us
            </Button>
            <IconButton
              aria-label="home"
              onClick={() => navigate("/")}
              sx={{
                color: colors.text,
                "&:hover": {
                  color: colors.primary,
                  transform: "scale(1.1)",
                  transition: "transform 0.2s",
                },
              }}
            >
              <HomeIcon fontSize="medium" />
            </IconButton>
          </Box>
        </Container>
      </Box>

      <Container maxWidth="lg" sx={{ py: 8 }}>
        <Typography
          variant="h3"
          sx={{
            mb: 6,
            textAlign: "center",
            background: `linear-gradient(45deg, ${colors.primary}, ${colors.secondary})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          How NeuroScan Works
        </Typography>

        <Grid container spacing={6}>
          {[
            {
              step: "1",
              title: "Upload Your MRI",
              description:
                "Simply drag and drop your DICOM files or select from your device",
              icon: "📤",
            },
            {
              step: "2",
              title: "AI Processing",
              description:
                "Our neural networks analyze 142 brain regions in under 60 seconds",
              icon: "⚡",
            },
            {
              step: "3",
              title: "Get Results",
              description:
                "Receive a detailed PDF report with highlighted areas of concern",
              icon: "📄",
            },
          ].map((item, index) => (
            <Grid item xs={12} md={4} key={index}>
              <Box
                sx={{
                  p: 4,
                  height: "100%",
                  backdropFilter: "blur(8px)",
                  background: "rgba(255, 255, 255, 0.08)",
                  border: `1px solid ${colors.primary}20`,
                  borderRadius: 4,
                  textAlign: "center",
                }}
              >
                <Typography variant="h2" sx={{ mb: 2 }}>
                  {item.icon}
                </Typography>
                <Typography
                  variant="h5"
                  sx={{ mb: 2, color: colors.primary, fontWeight: 600 }}
                >
                  {item.title}
                </Typography>
                <Typography variant="body1">{item.description}</Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}
