'use client'

import Image from "next/image";
import {useState} from "react";

export default function Home() {
  const [messages,setMessages] =useState([
    {
      role: "assistant",
      content: "Hi! I'm the Rate my Professor support assistant. How can i help you today?"
    }
   ])
  const [message,setMessage] = useState(' ')
  const sendMessage = async() =>{
    setMessages((messages)=>[
      ...messages,
      {role:"user",content:message},
      {role:"assistant",content:' '}
  ])

  const response = fetch('/api/chat',{
    method: "POST",
    headers:{
      'Content-Type': 'application/json'
    },
    body:JSON.stringify([...messages,{ role:"user",content:message}])
  })
  setMessage(' ')
  }

  
  return (
   
    );

}
