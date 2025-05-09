import React from "react";
import { useNavigate } from "react-router-dom";
import { Box, Button, Container, Grid, Typography } from "@mui/material";
import brainImage from "../assets/bg.jpg";
import logo from "../assets/brain.png";

const colors = {
  primary: "#00bcd4",
  secondary: "#ff4081",
  text: "#f5f5f5",
  darkBg: "#121212",
};

export default function LandingPage() {
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
        padding: "0 16px",
      }}
    >
      {/* Modified Navbar */}
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
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          {/* Logo on left */}
          <img src={logo} alt="Logo" style={{ height: 50 }} />

          {/* Navigation links on right */}
          <Box sx={{ display: "flex", gap: 4 }}>
            <Button
              color="inherit"
              sx={{
                color: colors.text,
                "&:hover": {
                  color: colors.primary,
                  transform: "translateY(-1px)",
                },
                transition: "all 0.2s",
              }}
              onClick={() => navigate("/how-it-works")}
            >
              How It Works
            </Button>
            <Button
              color="inherit"
              sx={{
                color: colors.text,
                "&:hover": {
                  color: colors.primary,
                  transform: "translateY(-1px)",
                },
                transition: "all 0.2s",
              }}
              onClick={() => navigate("/features")}
            >
              Features
            </Button>
            <Button
              color="inherit"
              sx={{
                color: colors.text,
                "&:hover": {
                  color: colors.primary,
                  transform: "translateY(-1px)",
                },
                transition: "all 0.2s",
              }}
              onClick={() => navigate("/about")}
            >
              About Us
            </Button>
          </Box>
        </Container>
      </Box>
      <Container maxWidth="md" sx={{ py: 6, textAlign: "center" }}>
        <Typography
          variant="h4"
          sx={{
            fontWeight: 700,
            mb: 2,
            background: `linear-gradient(45deg, ${colors.primary}, ${colors.secondary})`,
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            display: "inline-block",
          }}
        >
          NeuroScan AI
        </Typography>
        <Typography variant="subtitle1" sx={{ mb: 3, opacity: 0.9 }}>
          Precise neurological analysis in a compact interface
        </Typography>
        <Button
          variant="contained"
          size="medium"
          onClick={() => navigate("/home")}
          sx={{
            background: colors.primary,
            color: "white",
            padding: "8px 24px",
            fontSize: "0.875rem",
            "&:hover": {
              background: "#0097a7",
            },
          }}
        >
          Get Started
        </Button>
      </Container>

      <Container maxWidth="lg" sx={{ py: 4, pb: 8 }}>
        <Grid container spacing={3}>
          {[
            {
              title: "Tumor Detection",
              desc: "Early-stage identification",
              icon: "🔍",
              color: colors.primary,
            },
            {
              title: "Stroke Analysis",
              desc: "Predictive risk assessment",
              icon: "⚡",
              color: colors.secondary,
            },
            {
              title: "Progress Tracking",
              desc: "Treatment monitoring",
              icon: "📊",
              color: "#7c4dff",
            },
          ].map((item, index) => (
            <Grid item xs={12} sm={4} key={index}>
              <Box
                sx={{
                  p: 2,
                  height: "100%",
                  backdropFilter: "blur(8px)",
                  background: "rgba(255, 255, 255, 0.08)",
                  border: `1px solid ${item.color}20`,
                  borderRadius: 2,
                  transition: "all 0.2s",
                  "&:hover": {
                    borderColor: `${item.color}50`,
                  },
                }}
              >
                <Typography variant="h5" sx={{ textAlign: "center", mb: 1 }}>
                  {item.icon}
                </Typography>
                <Typography
                  variant="subtitle2"
                  sx={{
                    textAlign: "center",
                    fontWeight: 600,
                    mb: 0.5,
                    color: "white",
                  }}
                >
                  {item.title}
                </Typography>
                <Typography
                  variant="body2"
                  sx={{
                    textAlign: "center",
                    opacity: 0.8,
                    fontSize: "0.8rem",
                  }}
                >
                  {item.desc}
                </Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Container>
      <Box
        sx={{
          position: "fixed",
          bottom: 20,
          left: 0,
          right: 0,
          textAlign: "center",
          backgroundColor: "rgba(0,0,0,0.5)",
          color: "white",
          padding: "8px",
          zIndex: 1000,
          backdropFilter: "blur(2px)",
        }}
      >
        <Typography variant="subtitle2">Developed by Soumojit Ghosh</Typography>
      </Box>
    </Box>
  );
}
