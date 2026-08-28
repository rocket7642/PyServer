#include "asioConnector.h"

#include <iostream>
#include <asio.hpp>

using asio::ip::tcp;

TCPClient::TCPClient(const std::string& host, const std::string& service)
    : io_context_(), socket_(io_context_) {
    try {
        // Resolve the host and service to an endpoint
        tcp::resolver resolver(io_context_);
        tcp::resolver::results_type endpoints = resolver.resolve(host, service);

        // Connect to the server
        asio::connect(socket_, endpoints);
        std::cout << "Connected to server." << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Exception in TCPClient constructor: " << e.what() << std::endl;
    }
}

// Function to send data
void TCPClient::send_message(const std::string& message) {
    try {
        asio::write(socket_, asio::buffer(message));
        std::cout << "Sent: " << message << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Exception during send: " << e.what() << std::endl;
    }
}

std::string TCPClient::read_message() {
    try {
        if (socket_.available() == 0) {
            return "";  // No data available, return empty string
        }
        asio::streambuf buf;
        asio::read_until(socket_, buf, '\n');
        std::string data = asio::buffer_cast<const char*>(buf.data());
        return data;
    } catch (std::exception& e) {
        std::cerr << "Exception during read: " << e.what() << std::endl;
        return "";
    }
}

void TCPClient::disconnect() {
    asio::error_code ec;
    socket_.shutdown(tcp::socket::shutdown_both, ec);
    socket_.close();
}

// The socket's destructor will handle closing the connection implicitly
TCPClient::~TCPClient() {
    std::cout << "Client disconnected." << std::endl;
}