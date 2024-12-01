import React, { useState, useEffect } from "react";
import Card from "react-bootstrap/Card";

const LoadingText = () => {
  const [dots, setDots] = useState("");

  useEffect(() => {
    const interval = setInterval(() => {
      setDots((prev) => (prev.length < 3 ? prev + "." : ""));
    }, 500); // Update dots every 500ms

    return () => clearInterval(interval); 
  }, []);

  return (
    <Card.Text className="text-primary">
      Analyzing the image, please wait{dots}
    </Card.Text>
  );
};

export default LoadingText;
