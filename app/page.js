'use client'

import Image from "next/image";
import {useState } from "react";
import {Box, Button, TextField,Stack} from "@mui/material";

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
    body: JSON.stringify([...messages,{ role:"user",content:message}]),
  }).then(async(res)=> {
      const reader =res.body.getReader()
      const decoder = new TextDecoder()

      let result=''
      return reader.read().then(function processText({ done, value }) {
        if (done) {
          return result
        }
        const text = decoder.decode(value || new Uint8Array(), { stream: true })
        setMessages((messages)=>{
          let lastMessages = messages[messages.length - 1]
          let otherMessages = messages.slice(0, messages.length - 1)
          return[
            ...otherMessages,
            {...lastMessages, content: lastMessages.content+text},
          ]
        })

        return reader.read().then(processText)
      })
    })
  }

return (
<Box 
width="100vw" 
height="100vh" 
display="flex" 
justifyContent="center" 
alignItems="center">
  <Stack 
  direction="column" 
  width="500px" 
  height="700px" 
  border="1px solid black"
  p={2}
  spacing={3}
  >
    <Stack direction="column"
    spacing={2}
    flexGrow={1}
    overflow='auto'
    maxHeight='100%'
    >

    {messages.map((message,index)=>(
      <Box key={index} display="flex" 
      justifyContent={message.role=="assistant"?"flex-start":"flex-end"}
      >
      <Box bgcolor={message.role=="assistant"?"primary.main":"secondary.main"

      }
      color ="white"
      borderRadius={16}
      p={3}
      >
        {message.content}
        </Box>
      </Box>
    ))}
  </Stack>
  <Stack
  direction="row"
  spacing={2}>
    <TextField
    label="Message"
    fullWidth
    value={message}
    onChange={(e)=>{
      setMessage(e.target.value)
    }}/>
    <Button variant='contained' onClick={sendMessage}>Send</Button>
  </Stack>
  </Stack>
</Box>)

}
