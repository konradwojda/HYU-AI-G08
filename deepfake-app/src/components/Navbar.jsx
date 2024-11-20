function Navbar() {
    return (
        <header className="navbar">
            <img
            src="logo1.png" alt="Team Logo" className="logo"/>
            <nav id="navigaiton">
                <ul class="nav-links">
                    <li>Home</li>
                    <li><a href="https://github.com/konradwojda/HYU-AI-G08">About</a></li>
                </ul>
            </nav>

        </header>
    );
}

export default Navbar;