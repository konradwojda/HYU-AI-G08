function Navbar() {
    return (
        <header className="navbar">
            <a href="/"><img
            src="logo1.png" alt="Team Logo" className="logo" /></a>
            {/* <p>Deepfake Image Detector</p> */}
            <nav id="navigaiton">
                <ul class="nav-links">
                    <li><a href="/">Home</a></li>
                    <li><a href="https://github.com/konradwojda/HYU-AI-G08">About</a></li>
                </ul>
            </nav>

        </header>
    );
}

export default Navbar;