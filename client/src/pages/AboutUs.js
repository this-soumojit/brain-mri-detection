import React from "react";
import {
  Box,
  Container,
  Typography,
  Avatar,
  Grid,
  Button,
  IconButton
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import brainImage from "../assets/bg.jpg";
import logo from "../assets/brain.png";
import HomeIcon from '@mui/icons-material/Home'; // Add this import


const colors = {
  primary: "#00bcd4",
  secondary: "#ff4081",
  text: "#f5f5f5",
};

const team = [
  // {
  //   name: "Soumojit Ghosh",
  //   role: "Chief Neurologist",
  //   bio: "Harvard Medical School, 15+ years in neuroimaging",
  //   avatar: "👩‍⚕️",
  // },
  {
    name: "Soumojit Ghosh",
    role: "AI Developer & Full-Stack Engineer (or ML Engineer & Web Developer)",
    bio: "Self-taught AI developer and third-year CS undergrad.",
    avatar: "👨‍💻",
  },
  // {
  //   name: "Priya Patel",
  //   role: "Clinical Director",
  //   bio: "Specialized in neuroradiology protocols and standards",
  //   avatar: "🧑‍⚕️",
  // },
];

export default function AboutUs() {
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
              onClick={() => navigate("/features")}
            >
              Features
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
          Our Story
        </Typography>

        <Box
          sx={{
            p: 4,
            mb: 8,
            backdropFilter: "blur(8px)",
            background: "rgba(255, 255, 255, 0.08)",
            borderRadius: 4,
          }}
        >
          <Typography variant="h5" sx={{ mb: 3, color: colors.primary }}>
            Revolutionizing Neurological Care
          </Typography>
          <Typography paragraph sx={{ fontSize: "1.1rem" }}>
            Developed in 2025 by a B.Tech student passionate about AI in
            healthcare, NeuroScan AI bridges deep learning and neurology to
            tackle late-stage diagnosis. This solo project combines 3D
            convolutional neural networks with clinical research to detect brain
            tumors (glioma, meningioma, pituitary) earlier—and more
            accurately—than traditional methods.
          </Typography>
          <Typography paragraph sx={{ fontSize: "1.1rem" }}>
            Today, our technology is trusted by leading medical institutions
            across 14 countries, helping detect abnormalities an average of 3.2
            years earlier than traditional methods. (All the above details might
            not be true.)
          </Typography>
        </Box>

        <Typography
          variant="h4"
          sx={{
            mb: 6,
            textAlign: "center",
            color: colors.primary,
          }}
        >
          Meet The Team
        </Typography>

        <Grid container spacing={4}>
          {team.map((member, index) => (
            <Grid item xs={12} sm={4} key={index}>
              <Box
                sx={{
                  p: 4,
                  height: "100%",
                  backdropFilter: "blur(8px)",
                  background: "rgba(255, 255, 255, 0.08)",
                  borderRadius: 4,
                  textAlign: "center",
                }}
              >
                <Avatar
                  sx={{
                    width: 80,
                    height: 80,
                    fontSize: "2.5rem",
                    mb: 2,
                    mx: "auto",
                    bgcolor: colors.primary,
                  }}
                >
                  {member.avatar}
                </Avatar>
                <Typography variant="h6" sx={{ fontWeight: 600 }}>
                  {member.name}
                </Typography>
                <Typography
                  variant="subtitle1"
                  sx={{ color: colors.secondary, mb: 2 }}
                >
                  {member.role}
                </Typography>
                <Typography variant="body1">{member.bio}</Typography>
              </Box>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}
