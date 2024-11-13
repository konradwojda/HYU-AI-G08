import React from "react";
// import './App.css'
function Footer() {

  return (
    <footer className="footer">
      <section className="footer-content">
        <div className="footer-button">
          <a href="/">Home</a>
        </div>
        <div className="footer-button">
          <a href="/favorites">About</a>
        </div>
      </section>
      <p>Copyright &#169; 2024 TheFutureOfAI. All Rights Reserved</p>
    </footer>
  );
}

export default Footer;