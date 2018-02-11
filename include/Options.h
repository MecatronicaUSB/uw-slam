/**
* This file is part of UW-SLAM.
* 
* Copyright 2018.
* Developed by Fabio Morales <GitHub: /fmoralesh>,
*
* UW-SLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* You should have received a copy of the GNU General Public License
* along with UW-SLAM. If not, see <http://www.gnu.org/licenses/>.
*/

// Parse library
#include "args.hxx"

// Args declarations
args::ArgumentParser parser("Feature Detection Module.", "Author: Fabio Morales.");
args::HelpFlag help(parser, "help", "Display this help menu", {'h', "help"});

args::ValueFlag<std::string> dir_dataset(parser, "directory", "Directory of dataset files", {'d'});
args::ValueFlag<std::string> parse_calibration(parser, "calibration", "Name of input XML calibration file", {"calibration"});