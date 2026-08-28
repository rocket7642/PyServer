#ifndef ASIO_CONNECTOR_H
#define ASIO_CONNECTOR_H

#include <asio.hpp>
#include <string>

using asio::ip::tcp;

class TCPClient {
public:
    TCPClient(const std::string& host, const std::string& service);
    void send_message(const std::string& message);
    std::string read_message();
    void disconnect();
    ~TCPClient();

private:
    asio::io_context io_context_;
    tcp::socket socket_;
};

#endif // ASIO_CONNECTOR_H