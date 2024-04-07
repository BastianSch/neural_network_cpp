#include <iostream>
#include <string>

class InvalidShapeException : public std::exception
{
public:
    InvalidShapeException(const std::string& msg) : m_msg(msg)
    {
        std::cout << "MyException::MyException - set m_msg to:" << m_msg << std::endl;
    }

   ~InvalidShapeException()
   {
        std::cout << "MyException::~MyException" << std::endl;
   }

   virtual const char* what() const throw () 
   {
        std::cout << "InvalidShapeException::what" << std::endl;
        return m_msg.c_str();
   }

   const std::string m_msg;
};