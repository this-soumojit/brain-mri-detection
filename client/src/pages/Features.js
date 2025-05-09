import React from "react";
import {
  Box,
  Container,
  Typography,
  Grid,
  List,
  ListItem,
  ListItemText,
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

export default function Features() {
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
              onClick={() => navigate("/how-it-works")}
            >
              How It Works
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
          Advanced Features & Future Scope
        </Typography>

        <Grid container spacing={6}>
          <Grid item xs={12} md={6}>
            <Box
              sx={{
                p: 4,
                backdropFilter: "blur(8px)",
                background: "rgba(255, 255, 255, 0.08)",
                borderRadius: 4,
                height: "100%",
              }}
            >
              <Typography variant="h4" sx={{ mb: 3, color: colors.primary }}>
                Detection Capabilities
              </Typography>
              <List>
                {[
                  "Brain tumors (Gliomas, Meningiomas, Pituitary)",
                  "Ischemic and hemorrhagic strokes",
                  "Multiple Sclerosis lesions",
                  "Early Alzheimer's biomarkers",
                  "Parkinson's disease indicators",
                ].map((item, index) => (
                  <ListItem key={index} sx={{ py: 0.5 }}>
                    <ListItemText
                      primary={item}
                      primaryTypographyProps={{ style: { fontSize: "1.1rem" } }}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          </Grid>

          <Grid item xs={12} md={6}>
            <Box
              sx={{
                p: 4,
                backdropFilter: "blur(8px)",
                background: "rgba(255, 255, 255, 0.08)",
                borderRadius: 4,
                height: "100%",
              }}
            >
              <Typography variant="h4" sx={{ mb: 3, color: colors.secondary }}>
                Technical Specifications
              </Typography>
              <List>
                {[
                  "3D convolutional neural networks",
                  "DICOM, NIfTI, and JPEG2000 support",
                  "256-bit end-to-end encryption",
                  "HIPAA and GDPR compliant",
                  "API integration with PACS systems",
                ].map((item, index) => (
                  <ListItem key={index} sx={{ py: 0.5 }}>
                    <ListItemText
                      primary={item}
                      primaryTypographyProps={{ style: { fontSize: "1.1rem" } }}
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}
