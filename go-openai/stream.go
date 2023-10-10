package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
)

var (
	ErrTooManyEmptyStreamMessages = errors.New("stream has sent too many empty msgs")
)

type CompletionStream struct {
	emptyMessagesLimit uint
	isFinished         bool

	reader   *bufio.Reader
	response *http.Response
}

func (stream *CompletionStream) Recv() (response CompletionResponse, err error) {
	if stream.isFinished {
		err = io.EOF
		return
	}

	var emptyMessagesCount uint

waitForData:
	line, err := stream.reader.ReadBytes('\n')
	if err != nil {
		return
	}

	var headerData = []byte("data: ")
	line = bytes.TrimSpace(line)
	if !bytes.HasPrefix(line, headerData) {
		emptyMessagesCount++
		if emptyMessagesCount > stream.emptyMessagesLimit {
			err = ErrTooManyEmptyStreamMessages
			return
		}

		goto waitForData
	}

	line = bytes.TrimPrefix(line, headerData)
	if string(line) == "[DONE]" {
		stream.isFinished = true
		err = io.EOF
		return
	}

	err = json.Unmarshal(line, &response)
	return
}

func (stream *CompletionStream) Close() {
	stream.response.Body.Close()
}

func (c *Client) CreateCompletionStream(
	ctx context.Context,
	request CompletionRequest,
) (stream *CompletionStream, err error) {
	request.Stream = true
	req, err := c.newStreamRequest(ctx, "POST", "/completions", request)
	if err != nil {
		return
	}

	resp, err := c.config.HTTPClient.Do(req) 
	if err != nil {
		return
	}

	stream = &CompletionStream{
		emptyMessagesLimit: c.config.EmptyMessagesLimit,

		reader:   bufio.NewReader(resp.Body),
		response: resp,
	}
	return
}
