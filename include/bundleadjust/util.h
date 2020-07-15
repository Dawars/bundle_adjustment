double clamp(double val, double low, double high) {
    if (val > high) return high;
    else if (val < low) return low;
    else return val;
}

